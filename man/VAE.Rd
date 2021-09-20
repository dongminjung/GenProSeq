\name{VAE}
\alias{fit_VAE}
\alias{gen_VAE}

\title{
Variational autoencoder for generating protein sequences
}

\description{
The variational autoencoder (VAE) is a class of autoencoder where the encoder module is used to learn the parameter of a distribution and the decoder is used to generate examples from samples drawn from the learned distribution. The conditional variational autoencoder (CAVE) is designed to generate desired samples by including additional conditioning information. Since there may be underlying distinctions between groups of samples, the Gaussian mixture model is used for sequence generation. The Word2vec is applied to amino acids for embedding. The VAE or CVAE model can be trained by the function "fit_VAE", and then the function "gen_VAE" generates protein sequences from the trained model.
}

\usage{
fit_VAE(prot_seq,
        label = NULL,
        length_seq,
        embedding_dim,
        embedding_args = list(),
        latent_dim = 2,
        intermediate_encoder_layers,
        intermediate_decoder_layers,
        prot_seq_val = NULL,
        label_val = NULL,
        regularization = 1,
        epochs,
        batch_size,
        preprocessing = list(
            x_train = NULL,
            x_val = NULL,
            y_train = NULL,
            y_val = NULL,
            lenc = NULL,
            length_seq = NULL,
            num_seq = NULL,
            embedding_dim = NULL,
            embedding_matrix = NULL,
            removed_prot_seq = NULL,
            removed_prot_seq_val = NULL),
        use_generator = FALSE,
        optimizer = "adam",
        validation_split = 0, ...)

gen_VAE(x,
        label = NULL,
        num_seq,
        remove_gap = TRUE,
        batch_size,
        use_generator = FALSE)
}

\arguments{
    \item{prot_seq}{
aligned amino acid sequence
}
    \item{label}{
label (default: NULL)
}
    \item{length_seq}{
length of sequence
}
    \item{embedding_dim}{
dimension of the dense embedding
}
    \item{embedding_args}{
list of arguments for "word2vec::word2vec" but for dim, min_count and split
}
    \item{latent_dim}{
dimension of latent vector (default: 2)
}
    \item{intermediate_encoder_layers}{
list of intermediate layers for encoder, without input layer
}
    \item{intermediate_decoder_layers}{
list of intermediate layers for decoder, without output layer
}
    \item{regularization}{
regularization parameter, which is nonnegative (default: 1)
}
    \item{prot_seq_val}{
amino acid sequence for validation (default: NULL)
}
    \item{label_val}{
label for validation (default: NULL)
}
    \item{epochs}{
number of epochs
}
    \item{batch_size}{
batch size
}
    \item{preprocessing}{
list of preprocessed results, they are set to NULL as default
\itemize{
    \item x_train : embedded sequence data for train
    \item x_val : embedded sequence data for validation
    \item y_train : labels for train
    \item y_val : labels for validation
    \item lenc : encoded labels
    \item length_seq : length of sequence
    \item num_seq : number of sequences for train
    \item embedding_dim : dimension of the dense embedding
    \item embedding_matrix : embedding matrix
    \item removed_prot_seq : index for removed protein sequences while checking
    \item removed_prot_seq_val : index for removed protein sequences of validation
}
}
    \item{use_generator}{
use data generator if TRUE (default: FALSE)
}
    \item{optimizer}{
name of optimizer (default: adam)
}
    \item{validation_split}{
proportion of validation data, it is ignored when there is a validation set (default: 0)
}
    \item{...}{
additional parameters for the "fit"
}
    \item{x}{
result of the function "fit_VAE"
}
    \item{num_seq}{
number of sequences to be generated
}
    \item{remove_gap}{
remove gaps from sequences (default: TRUE)
}
}


\value{
    \item{model}{
trained VAE model
}
    \item{encoder}{
trained encoder model
}
    \item{decoder}{
trained decoder model
}
    \item{preprocessing}{
preprocessed results
}
    \item{gen_seq}{
generated sequence data
}
    \item{label}{
labels for generated sequence data
}
    \item{latent_vector}{
latent vector from embedded sequence data
}
}


\author{
Dongmin Jung
}


\references{
Cinelli, L. P., Marins, M. A., da Silva, E. A. B., & Netto, S. L. (2021). Variational Methods for Machine Learning with Applications to Deep Networks. Springer.

Liebowitz, J. (Ed.). (2020). Data Analytics and AI. CRC Press.
}


\seealso{
keras::fit, keras::compile, reticulate::array_reshape, mclust::mclustBIC, mclust::mclustModel, mclust::sim, DeepPINCS::multiple_sampling_generator, CatEncoders::LabelEncoder.fit, CatEncoders::transform, CatEncoders::inverse.transform
}


\examples{
label <- substr(example_luxA, 3, 3)

# model parameters
length_seq <- 360
embedding_dim <- 8
batch_size <- 128
epochs <- 2

# CVAE
VAE_result <- fit_VAE(prot_seq = example_luxA,
                    label = label,
                    length_seq = length_seq,
                    embedding_dim = embedding_dim,
                    embedding_args = list(iter = 20),
                    intermediate_encoder_layers = list(layer_dense(units = 128),
                                                        layer_dense(units = 16)),
                    intermediate_decoder_layers = list(layer_dense(units = 16),
                                                        layer_dense(units = 128)),
                    prot_seq_val = example_luxA,
                    label_val = label,
                    epochs = epochs,
                    batch_size = batch_size,
                    use_generator = FALSE,
                    optimizer = keras::optimizer_adam(clipnorm = 0.1),
                    callbacks = keras::callback_early_stopping(
                        monitor = "val_loss",
                        patience = 10,
                        restore_best_weights = TRUE))
gen_prot_VAE_I <- gen_VAE(VAE_result, label = rep("I", 100), num_seq = 100)
gen_prot_VAE_L <- gen_VAE(VAE_result, label = rep("L", 100), num_seq = 100)


### from preprocessing
VAE_result2 <- fit_VAE(intermediate_encoder_layers = list(layer_dense(units = 128),
                                                        layer_dense(units = 16)),
                        intermediate_decoder_layers = list(layer_dense(units = 16),
                                                        layer_dense(units = 128)),
                        epochs = epochs, batch_size = batch_size,
                        preprocessing = VAE_result$preprocessing,
                        use_generator = FALSE,
                        optimizer = keras::optimizer_adam(clipnorm = 0.1),
                        callbacks = keras::callback_early_stopping(
                            monitor = "val_loss",
                            patience = 10,
                            restore_best_weights = TRUE))
gen_prot_VAE2_I <- gen_VAE(VAE_result2, label = rep("I", 100), num_seq = 100)
gen_prot_VAE2_L <- gen_VAE(VAE_result2, label = rep("L", 100), num_seq = 100)
}
