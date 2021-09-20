# model parameters
length_seq <- 403
embedding_dim <- 8
latent_dim <- 4
epochs <- 20
batch_size <- 64



test_that("fit_GAN: miss training data", {
    expect_error(
        fit_GAN(length_seq = length_seq,
                embedding_dim = embedding_dim,
                latent_dim = latent_dim,
                intermediate_generator_layers = list(
                    layer_dense(units = 16),
                    layer_dense(units = 128)),
                intermediate_discriminator_layers = list(
                    layer_dense(units = 128, activation = "relu"),
                    layer_dense(units = 16, activation = "relu")),
                prot_seq_val = example_PTEN,
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_GAN: miss length of sequence", {
    expect_error(
        fit_GAN(prot_seq = example_PTEN,
                embedding_dim = embedding_dim,
                latent_dim = latent_dim,
                intermediate_generator_layers = list(
                    layer_dense(units = 16),
                    layer_dense(units = 128)),
                intermediate_discriminator_layers = list(
                    layer_dense(units = 128, activation = "relu"),
                    layer_dense(units = 16, activation = "relu")),
                prot_seq_val = example_PTEN,
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_GAN: miss embedding dimension", {
    expect_error(
        fit_GAN(prot_seq = example_PTEN,
                length_seq = length_seq,
                latent_dim = latent_dim,
                intermediate_generator_layers = list(
                    layer_dense(units = 16),
                    layer_dense(units = 128)),
                intermediate_discriminator_layers = list(
                    layer_dense(units = 128, activation = "relu"),
                    layer_dense(units = 16, activation = "relu")),
                prot_seq_val = example_PTEN,
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_GAN: miss latent dimension", {
    expect_error(
        fit_GAN(prot_seq = example_PTEN,
                length_seq = length_seq,
                embedding_dim = embedding_dim,
                intermediate_generator_layers = list(
                    layer_dense(units = 16),
                    layer_dense(units = 128)),
                intermediate_discriminator_layers = list(
                    layer_dense(units = 128, activation = "relu"),
                    layer_dense(units = 16, activation = "relu")),
                prot_seq_val = example_PTEN,
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_GAN: miss intermediate generator layers", {
    expect_error(
        fit_GAN(prot_seq = example_PTEN,
                length_seq = length_seq,
                embedding_dim = embedding_dim,
                latent_dim = latent_dim,
                intermediate_discriminator_layers = list(
                    layer_dense(units = 128, activation = "relu"),
                    layer_dense(units = 16, activation = "relu")),
                prot_seq_val = example_PTEN,
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_GAN: miss intermediate discriminator layers", {
    expect_error(
        fit_GAN(prot_seq = example_PTEN,
                length_seq = length_seq,
                embedding_dim = embedding_dim,
                latent_dim = latent_dim,
                intermediate_generator_layers = list(
                    layer_dense(units = 16),
                    layer_dense(units = 128)),
                prot_seq_val = example_PTEN,
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_GAN: miss epochs", {
    expect_error(
        fit_GAN(prot_seq = example_PTEN,
                length_seq = length_seq,
                embedding_dim = embedding_dim,
                latent_dim = latent_dim,
                intermediate_generator_layers = list(
                    layer_dense(units = 16),
                    layer_dense(units = 128)),
                intermediate_discriminator_layers = list(
                    layer_dense(units = 128, activation = "relu"),
                    layer_dense(units = 16, activation = "relu")),
                prot_seq_val = example_PTEN,
                batch_size = batch_size)
    )
})



test_that("fit_GAN: miss batch size", {
    expect_error(
        fit_GAN(prot_seq = example_PTEN,
                length_seq = length_seq,
                embedding_dim = embedding_dim,
                latent_dim = latent_dim,
                intermediate_generator_layers = list(
                    layer_dense(units = 16),
                    layer_dense(units = 128)),
                intermediate_discriminator_layers = list(
                    layer_dense(units = 128, activation = "relu"),
                    layer_dense(units = 16, activation = "relu")),
                prot_seq_val = example_PTEN,
                epochs = epochs)
    )
})



test_that("gen_GAN: miss result of fit_GAN", {
    expect_error(
        gen_GAN(num_seq = 100)
    )
})



test_that("fit_VAE: miss training data", {
    expect_error(
        fit_VAE(length_seq = length_seq,
                embedding_dim = embedding_dim,
                latent_dim = latent_dim,
                intermediate_encoder_layers = list(
                    layer_dense(units = 16),
                    layer_dense(units = 128)),
                intermediate_decoder_layers = list(
                    layer_dense(units = 128, activation = "relu"),
                    layer_dense(units = 16, activation = "relu")),
                prot_seq_val = example_PTEN,
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_VAE: miss length of sequence", {
    expect_error(
        fit_VAE(prot_seq = example_PTEN,
                embedding_dim = embedding_dim,
                latent_dim = latent_dim,
                intermediate_encoder_layers = list(
                    layer_dense(units = 16),
                    layer_dense(units = 128)),
                intermediate_decoder_layers = list(
                    layer_dense(units = 128, activation = "relu"),
                    layer_dense(units = 16, activation = "relu")),
                prot_seq_val = example_PTEN,
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_VAE: miss embedding dimension", {
    expect_error(
        fit_VAE(prot_seq = example_PTEN,
                length_seq = length_seq,
                latent_dim = latent_dim,
                intermediate_encoder_layers = list(
                    layer_dense(units = 16),
                    layer_dense(units = 128)),
                intermediate_decoder_layers = list(
                    layer_dense(units = 128, activation = "relu"),
                    layer_dense(units = 16, activation = "relu")),
                prot_seq_val = example_PTEN,
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_VAE: miss intermediate decoder layers", {
    expect_error(
        fit_VAE(prot_seq = example_PTEN,
                length_seq = length_seq,
                embedding_dim = embedding_dim,
                latent_dim = latent_dim,
                intermediate_encoder_layers = list(
                    layer_dense(units = 128, activation = "relu"),
                    layer_dense(units = 16, activation = "relu")),
                prot_seq_val = example_PTEN,
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_VAE: miss intermediate encoder layers", {
    expect_error(
        fit_VAE(prot_seq = example_PTEN,
                length_seq = length_seq,
                embedding_dim = embedding_dim,
                latent_dim = latent_dim,
                intermediate_decoder_layers = list(
                    layer_dense(units = 16),
                    layer_dense(units = 128)),
                prot_seq_val = example_PTEN,
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_VAE: miss epochs", {
    expect_error(
        fit_VAE(prot_seq = example_PTEN,
                length_seq = length_seq,
                embedding_dim = embedding_dim,
                latent_dim = latent_dim,
                intermediate_encoder_layers = list(
                    layer_dense(units = 16),
                    layer_dense(units = 128)),
                intermediate_decoder_layers = list(
                    layer_dense(units = 128, activation = "relu"),
                    layer_dense(units = 16, activation = "relu")),
                prot_seq_val = example_PTEN,
                batch_size = batch_size)
    )
})



test_that("fit_VAE: miss batch size", {
    expect_error(
      fit_VAE(prot_seq = example_PTEN,
              length_seq = length_seq,
              embedding_dim = embedding_dim,
              latent_dim = latent_dim,
              intermediate_encoder_layers = list(
                  layer_dense(units = 16),
                  layer_dense(units = 128)),
              intermediate_decoder_layers = list(
                  layer_dense(units = 128, activation = "relu"),
                  layer_dense(units = 16, activation = "relu")),
              prot_seq_val = example_PTEN,
              epochs = epochs)
    )
})



test_that("gen_VAE: miss result of fit_VAE", {
    expect_error(
        gen_VAE(num_seq = 100)
    )
})



length_seq <- 10
num_heads <- 2
ff_dim <- 16
num_transformer_blocks <- 2



test_that("fit_ART: miss training data", {
    expect_error(
        fit_ART(length_seq = length_seq,
                embedding_dim = embedding_dim,
                num_heads = num_heads,
                ff_dim = ff_dim,
                num_transformer_blocks = num_transformer_blocks,
                layers = list(layer_dropout(rate = 0.1),
                              layer_dense(units = 32, activation = "relu"),
                              layer_dropout(rate = 0.1)),
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_ART: invalid length of sequence", {
    expect_error(
        fit_ART(prot_seq = example_PTEN[1:5],
                length_seq = 500,
                embedding_dim = embedding_dim,
                num_heads = num_heads,
                ff_dim = ff_dim,
                num_transformer_blocks = num_transformer_blocks,
                layers = list(layer_dropout(rate = 0.1),
                              layer_dense(units = 32, activation = "relu"),
                              layer_dropout(rate = 0.1)),
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_ART: miss embedding dimension", {
    expect_error(
        fit_ART(prot_seq = example_PTEN[1:5],
                length_seq = length_seq,
                num_heads = num_heads,
                ff_dim = ff_dim,
                num_transformer_blocks = num_transformer_blocks,
                layers = list(layer_dropout(rate = 0.1),
                              layer_dense(units = 32, activation = "relu"),
                              layer_dropout(rate = 0.1)),
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_ART: miss num_heads", {
    expect_error(
        fit_ART(prot_seq = example_PTEN[1:5],
                length_seq = length_seq,
                embedding_dim = embedding_dim,
                ff_dim = ff_dim,
                num_transformer_blocks = num_transformer_blocks,
                layers = list(layer_dropout(rate = 0.1),
                              layer_dense(units = 32, activation = "relu"),
                              layer_dropout(rate = 0.1)),
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_ART: miss ff_dim", {
    expect_error(
        fit_ART(prot_seq = example_PTEN[1:5],
                length_seq = length_seq,
                num_heads = num_heads,
                embedding_dim = embedding_dim,
                num_transformer_blocks = num_transformer_blocks,
                layers = list(layer_dropout(rate = 0.1),
                              layer_dense(units = 32, activation = "relu"),
                              layer_dropout(rate = 0.1)),
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_ART: miss num_transformer_blocks", {
    expect_error(
        fit_ART(prot_seq = example_PTEN[1:5],
                length_seq = length_seq,
                num_heads = num_heads,
                embedding_dim = embedding_dim,
                ff_dim = ff_dim,
                layers = list(layer_dropout(rate = 0.1),
                              layer_dense(units = 32, activation = "relu"),
                              layer_dropout(rate = 0.1)),
                epochs = epochs,
                batch_size = batch_size)
    )
})



test_that("fit_ART: miss epochs", {
    expect_error(
        fit_ART(prot_seq = example_PTEN[1:5],
                length_seq = length_seq,
                num_heads = num_heads,
                embedding_dim = embedding_dim,
                ff_dim = ff_dim,
                num_transformer_blocks = num_transformer_blocks,
                layers = list(layer_dropout(rate = 0.1),
                              layer_dense(units = 32, activation = "relu"),
                              layer_dropout(rate = 0.1)),
                batch_size = batch_size)
    )
})



test_that("fit_ART: miss batch size", {
    expect_error(
        fit_ART(prot_seq = example_PTEN[1:5],
                length_seq = length_seq,
                num_heads = num_heads,
                embedding_dim = embedding_dim,
                ff_dim = ff_dim,
                num_transformer_blocks = num_transformer_blocks,
                layers = list(layer_dropout(rate = 0.1),
                              layer_dense(units = 32, activation = "relu"),
                              layer_dropout(rate = 0.1)),
                epochs = epochs)
    )
})



test_that("gen_ART: miss result of fit_ART", {
    expect_error(
        gen_ART(seed_prot = "SGFRKMAFPS", length_AA = 20, method = "greedy")
    )
})
