"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn
EGCN with batch processing
"""
import os
os.environ["MXNET_USE_OPERATOR_TUNING"] = "0"
import argparse
import numpy as np
import time
import mxnet as mx
from mxnet import gluon
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

class EGCNLayer(gluon.Block):
    def __init__(self,
                 g,
                 out_feats,
                 activation,
                 dropout):
        super(EGCNLayer, self).__init__()
        self.g = g
        self.dense = gluon.nn.Dense(out_feats, activation)
        self.edense = gluon.nn.Dense(out_feats, activation)
        self.dropout = dropout


    def forward(self, h):
        self.g.ndata['h'] = h * self.g.ndata['out_norm']
        # somewhat hacky, who cares
        egcn_message = lambda edges: {'m' : mx.nd.concat(mx.nd.Dropout(self.edense(mx.nd.concat(edges.dst['h'], edges.src['h'], dim=1)), p=self.dropout), edges.src['h'], dim=1)}
        self.g.update_all(egcn_message, 
                          fn.sum(msg='m', out='accum'))
        accum = self.g.ndata.pop('accum')
        accum = self.dense(accum * self.g.ndata['in_norm'])
        if self.dropout:
            accum = mx.nd.Dropout(accum, p=self.dropout)
        h = self.g.ndata.pop('h')
        h = mx.nd.concat(h / self.g.ndata['out_norm'], accum, dim=1)
        return h


class EGCN(gluon.Block):
    def __init__(self,
                 g,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(EGCN, self).__init__()
        self.inp_layer = gluon.nn.Dense(n_hidden, activation)
        self.dropout = dropout
        self.layers = gluon.nn.Sequential()
        for i in range(n_layers):
            self.layers.add(EGCNLayer(g, n_hidden, activation, dropout))
        self.out_layer = gluon.nn.Dense(n_classes)


    def forward(self, features):
        emb_inp = [features, self.inp_layer(features)]
        if self.dropout:
            emb_inp[-1] = mx.nd.Dropout(emb_inp[-1], p=self.dropout)
        h = mx.nd.concat(*emb_inp, dim=1)
        for layer in self.layers:
            h = layer(h)
        h = self.out_layer(h)
        return h


def evaluate(model, features, labels, mask):
    pred = model(features)
    loss_fcn = gluon.loss.SoftmaxCELoss()
    loss = loss_fcn(pred, labels, mx.nd.expand_dims(mask, 1))
    loss = loss.sum() / mask.sum().asscalar()
    accuracy = ((pred.argmax(axis=1) == labels) * mask).sum() / mask.sum().asscalar()
    return loss.asscalar(), accuracy.asscalar()


def main(args):
    # load and preprocess dataset
    data = load_data(args)

    if args.self_loop:
        data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

    features = mx.nd.array(data.features)
    labels = mx.nd.array(data.labels)
    train_mask = mx.nd.array(data.train_mask)
    val_mask = mx.nd.array(data.val_mask)
    test_mask = mx.nd.array(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.sum().asscalar(),
              val_mask.sum().asscalar(),
              test_mask.sum().asscalar()), flush=True)

    if args.gpu < 0:
        cuda = False
        ctx = mx.cpu(0)
    else:
        cuda = True
        ctx = mx.gpu(args.gpu)

    features = features.as_in_context(ctx)
    labels = labels.as_in_context(ctx)
    train_mask = train_mask.as_in_context(ctx)
    val_mask = val_mask.as_in_context(ctx)
    test_mask = test_mask.as_in_context(ctx)

    # create GCN model
    g = DGLGraph(data.graph)
    # normalization
    in_degs = g.in_degrees().astype('float32')
    out_degs = g.out_degrees().astype('float32')
    in_norm = mx.nd.power(in_degs, -0.5)
    out_norm = mx.nd.power(out_degs, -0.5)
    if cuda:
        in_norm = in_norm.as_in_context(ctx)
        out_norm = out_norm.as_in_context(ctx)
    g.ndata['in_norm'] = mx.nd.expand_dims(in_norm, 1)
    g.ndata['out_norm'] = mx.nd.expand_dims(out_norm, 1)

    model = EGCN(g,
                args.n_hidden,
                n_classes,
                args.n_layers,
                'relu',
                args.dropout,
                )
    model.initialize(mx.init.MSRAPrelu(), ctx=ctx)
    # model.initialize(mx.init.Xavier(), ctx=ctx)
    n_train_samples = train_mask.sum().asscalar()
    loss_fcn = gluon.loss.SoftmaxCELoss()

    # use optimizer
    print(model.collect_params(), flush=True)
    trainer = gluon.Trainer(model.collect_params(), 'adam',
            {'learning_rate': args.lr, 'wd': args.weight_decay})

    # initialize graph
    dur = []
    best_val_loss = 999999
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()
        # forward
        with mx.autograd.record():
            pred = model(features)
            loss = loss_fcn(pred, labels, mx.nd.expand_dims(train_mask, 1))
            loss = loss.sum() / n_train_samples

        loss.backward()
        trainer.step(batch_size=1)

        if epoch >= 3:
            dur.append(time.time() - t0)
            val_loss, val_acc = evaluate(model, features, labels, val_mask)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_parameters(args.save)
            print("Epoch {:05d} | Time(s) {:.4f} | Val loss {:.4f} | Val accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}". format(
                epoch, np.mean(dur), val_loss, val_acc, n_edges / np.mean(dur) / 1000), flush=True)

    # test set accuracy
    model.load_parameters(args.save)
    test_loss, test_acc = evaluate(model, features, labels, test_mask)
    print("Test loss {:.4f} | Test accuracy {:.2%}".format(test_loss, test_acc), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EGCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--normalization",
            choices=['sym','left'], default=None,
            help="graph normalization types (default=None)")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--save", type=str,
            help="path for the best model")            
    args = parser.parse_args()

    print(args, flush=True)

    main(args)
