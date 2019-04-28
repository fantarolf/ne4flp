import os

import keras

from ..emb import NodeEmbeddings
from ..emb import lp_autoencoder as lpae
from ..conf import SCORE_FUNC, SCORE_FUNC_CLF
from ..exp.emb import tune_clf_emb
from ..util import write_pkl, write_score, save_embedding

jp = os.path.join
# TODO scoring as fct. arg?


def lp_autoencoder_experiment(data,
                              d,
                              ae_layers,
                              lp_layers,
                              l1_reg,
                              l2_reg,
                              beta,
                              alpha,
                              batch_size,
                              pretrain_epochs,
                              train_epochs,
                              out_path):
    """ Calculate test score for lp autoencoder on dataset

    Parameters
    ----------
    data:
        D_test of the graph (as returned by `data.read_in_splits(graph)[1]')
    d : int
        dimension of embedding
    ae_layers : list of ints
        dimension of autoencoder layers (except middle layer that is controlled
        by `d`.
    lp_layers : list of ints
        dimension of link-predictor layers
    l1_reg : float
        weight for l1 regularization
    l2_reg : float
        weight for l2 regularization
    beta : float
        beta- control parameter for SDNE 2nd order loss
    alpha : float
        weight for reconstruction loss
    batch_size : int
        size of data batches
    pretrain_epochs : int
        number of pretrain epochs
    train_epochs : int
        max. number of train epochs (usually less as early stopping is used)
    out_path : str
        where to store the results. Must be non-existing or an empty-directory

    Returns
    -------

    """
    data_path = jp(out_path, 'data')
    os.mkdir(data_path)

    G, pe, ne = data
    nV = G.number_of_nodes()

    # construct autoencoder
    encoder = lpae.get_encoder(nV, d, ae_layers, l1_reg, l2_reg)
    decoder = lpae.get_decoder(nV, d, ae_layers[::-1], l1_reg, l2_reg)
    autoencoder = lpae.get_autoencoder(encoder, decoder)

    pretrain_ae_in = keras.layers.Input((nV,))
    encoder_out, decoder_out = autoencoder(pretrain_ae_in)
    pretrain_ae = keras.models.Model(inputs=pretrain_ae_in, outputs=decoder_out)

    pretrain_ae.compile(
        loss=lpae.sdne_2_loss(beta),
        optimizer='adadelta',
        metrics=[lpae.sdne_2_loss(beta)]  # loss w/o regularization
    )

    pretrain_generator = lpae.AutoencoderGenerator(G, batch_size)

    # pretrain model
    pretrain_hist = pretrain_ae.fit_generator(
        pretrain_generator,
        epochs=pretrain_epochs
    )

    # save pretrain history
    pretrain_hist_path = jp(out_path, 'pretrain-history.pkl')
    write_pkl(pretrain_hist_path, pretrain_hist.history)

    # get link predictor and construct full model
    link_predictor = lpae.get_link_predictor(d, lp_layers, l1_reg, l2_reg)

    full_model = lpae.get_lp_autoencoder(autoencoder, link_predictor)

    pred_model = keras.models.Model(
        inputs=full_model.inputs,
        outputs=full_model.outputs[0]
    )

    # calculate input graphs
    G_val = G.copy()
    # add train edges to val input (as described in Sec 5.3.2)
    G_val.add_edges_from(pe[0])
    G_test = G.copy()

    graphs = [G, G_val, G_test]

    # generators for train, val, and test inp. and outp.
    generators = [
        lpae.LPAutoencoderGenerator(g, p, n, batch_size) for (g, p, n) in zip(
            graphs, pe, ne)
    ]

    full_model.compile(
        loss=['binary_crossentropy',
              lpae.sdne_2_loss(beta),
              lpae.sdne_2_loss(beta)],
        optimizer='adadelta',
        loss_weights=[1, alpha, alpha],
    )

    # callback to save encoded and decoded data every 5 epochs (not used)
    train_callback = TrainCallback(pretrain_generator,
                                   encoder=encoder,
                                   autoencoder=autoencoder,
                                   out_path=data_path)

    # early stopping monitored by val loss
    early_stopping = keras.callbacks.EarlyStopping(
        patience=2,
        monitor='val_loss',
        restore_best_weights=True
    )

    # fit model
    full_model_hist = full_model.fit_generator(
        generators[0],
        validation_data=generators[1],
        epochs=train_epochs,
        callbacks=[train_callback,
                   early_stopping]
    )

    # save history
    full_model_hist_path = jp(out_path, 'full-model-history.pkl')
    write_pkl(full_model_hist_path, full_model_hist.history)

    # save model weights
    full_model_weights_path = jp(out_path, 'full-model-weights.h5')
    full_model.save_weights(full_model_weights_path)

    # save model configs
    full_model_path = jp(out_path, 'full-model.yaml')
    with open(full_model_path, 'w') as f:
        f.write(full_model.to_yaml())

    # G with updated edges (train and val edges)
    G_more_edges = G.copy()
    G_more_edges.add_edges_from(pe[0] + pe[1])

    # score paths for scores reported in tab. 12
    score_paths = [
        jp(out_path, 'trained-score.txt'),
        jp(out_path, 'trained-score-more-edges.txt'),
        jp(out_path, 'trained-score-train-edges.txt')
    ]

    # data generators for batchwise prediction
    pred_generators = [
        (generators[2], pretrain_generator),
        (lpae.LPAutoencoderGenerator(G_more_edges, pe[2], ne[2], batch_size),
         lpae.AutoencoderGenerator(G_more_edges, batch_size)),
        (generators[0], None)
    ]

    # predict from input data and write scores to disk
    for path, (lp_gen, emb_gen) in zip(score_paths, pred_generators):

        if not os.path.exists(path):
            true_links = lp_gen.all_lp_targets()
            pred_links = pred_model.predict_generator(lp_gen)
            lpae_score = SCORE_FUNC(true_links, pred_links)

            write_score(path, lpae_score)

        name, ext = os.path.splitext(path)
        path_clf = f'{name}-clf{ext}'
        if not os.path.exists(path_clf) and emb_gen is not None:
            embeddings = encoder.predict_generator(emb_gen)
            emb = NodeEmbeddings.from_array(emb_gen.G, embeddings)
            score_clf, _ = tune_clf_emb(emb, pe, ne, SCORE_FUNC_CLF)
            write_score(path_clf, score_clf)


def lp_autoencoder_experiment_tuning(data,
                                     d,
                                     ae_layers,
                                     lp_layers,
                                     l1_reg,
                                     l2_reg,
                                     beta,
                                     alpha,
                                     batch_size,
                                     pretrain_epochs,
                                     train_epochs):
    """Experiment for hyperparameter search, works mostly as above,
    but does not store as much information. Early stopping is not used.
    Returns loss histories and obtained score
    """
    G, pe, ne = data
    nV = G.number_of_nodes()

    encoder = lpae.get_encoder(nV, d, ae_layers, l1_reg, l2_reg)
    decoder = lpae.get_decoder(nV, d, ae_layers[::-1], l1_reg, l2_reg)
    autoencoder = lpae.get_autoencoder(encoder, decoder)

    pretrain_ae_in = keras.layers.Input((nV,))
    encoder_out, decoder_out = autoencoder(pretrain_ae_in)
    pretrain_ae = keras.models.Model(inputs=pretrain_ae_in, outputs=decoder_out)

    pretrain_ae.compile(
        loss=lpae.sdne_2_loss(beta),
        optimizer='adadelta',
        metrics=[lpae.sdne_2_loss(beta)]  # loss w/o regularization
    )

    pretrain_generator = lpae.AutoencoderGenerator(G, batch_size)

    pretrain_hist = pretrain_ae.fit_generator(
        pretrain_generator,
        epochs=pretrain_epochs,
    )

    link_predictor = lpae.get_link_predictor(d, lp_layers, l1_reg, l2_reg)

    full_model = lpae.get_lp_autoencoder(autoencoder, link_predictor)

    pred_model = keras.models.Model(
        inputs=full_model.inputs,
        outputs=full_model.outputs[0]
    )

    G_val = G.copy()
    G_val.add_edges_from(pe[0])
    G_test = G.copy()

    graphs = [G, G_val, G_test]

    generators = [
        lpae.LPAutoencoderGenerator(g, p, n, batch_size) for (g, p, n) in zip(
            graphs, pe, ne)
    ]

    full_model.compile(
        loss=['binary_crossentropy',
              lpae.sdne_2_loss(beta),
              lpae.sdne_2_loss(beta)],
        optimizer='adadelta',
        loss_weights=[1, alpha, alpha],
    )

    full_model_hist = full_model.fit_generator(
        generators[0],
        validation_data=generators[1],
        epochs=train_epochs
    )

    true_links = generators[2].all_lp_targets()
    pred_links = pred_model.predict_generator(generators[2])

    trained_score = SCORE_FUNC(true_links, pred_links)

    hists = pretrain_hist.history, full_model_hist.history

    return hists, trained_score


class TrainCallback(keras.callbacks.Callback):
    """save embeddings and decoded"""

    def __init__(self,
                 data_generator,
                 encoder,
                 autoencoder,
                 out_path,
                 print_step=5):
        super().__init__()
        self.data_generator = data_generator
        self.print_step = print_step
        self.out_path = out_path
        self.encoder = encoder
        self.autoencoder = autoencoder

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.print_step == 0:
            A = self.data_generator.all_data()
            # store decoded A
            A_decoded = self.autoencoder.predict(A)[1]
            A_decoded_path = jp(self.out_path,
                                f'train_decoded_{epoch}.npy')
            save_embedding(A_decoded, True, A_decoded_path)
            # store encoded A
            A_encoded = self.encoder.predict(A)
            A_encoded_path = jp(self.out_path,
                                f'train_encoded_{epoch}.npy')
            save_embedding(A_encoded, False, A_encoded_path)


class PretrainCallback(keras.callbacks.Callback):
    """save embeddings and decoded A"""

    def __init__(self, data_generator, encoder, out_path, print_step=5):
        super().__init__()
        self.data_generator = data_generator
        self.print_step = print_step
        self.out_path = out_path
        self.encoder = encoder

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.print_step == 0:
            A = self.data_generator.all_data()
            # store decoded A
            A_decoded = self.model.predict(A)
            A_decoded_path = jp(self.out_path,
                                f'pretrain_decoded_{epoch}.npy')
            save_embedding(A_decoded, True, A_decoded_path)
            # store encoded A
            A_encoded = self.encoder.predict(A)
            A_encoded_path = jp(self.out_path,
                                f'pretrain_encoded_{epoch}.npy')
            save_embedding(A_encoded, False, A_encoded_path)
