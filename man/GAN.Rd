\name{GAN}
\alias{fit_GAN}
\alias{gen_GAN}

\title{
Generative adversarial network for generating protein sequences
}

\description{
The generative adversarial network (GAN) is made up of a discriminator and a generator that compete in a two-player minimax game. The objective of the generator is to produce an output that is so close to real that it confuses the discriminator in being able to differentiate the fake data from the real data. The conditional GAN (CGAN) is based on vanilla GAN with additional conditional input to generator and discriminator. The auxiliary classifier GAN (ACGAN) is an extension of CGAN that adds conditional input only to the generator. The Word2vec is applied to amino acids for embedding. The GAN or ACGAN model can be trained by the function "fit_GAN", and then the function "gen_GAN" generates protein sequences from the trained model.
}

\usage{
fit_GAN(prot_seq,
        label = NULL,
        length_seq,
        embedding_dim,
        embedding_args = list(),
        latent_dim = NULL,
        intermediate_generator_layers,
        intermediate_discriminator_layers,
        prot_seq_val = NULL,
        label_val = NULL,
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
            removed_prot_seq_val = NULL,
            latent_dim = NULL),
        optimizer = "adam",
        validation_split = 0)

gen_GAN(x,
        label = NULL,
        num_seq,
        remove_gap = TRUE)
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
dimension of latent vector (default: NULL)
}
    \item{intermediate_generator_layers}{
list of intermediate layers for generator, without input layer
}
    \item{intermediate_discriminator_layers}{
list of intermediate layers for discriminator, without output layer
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
    \item latent_dim : dimension of latent vector
}
}
    \item{optimizer}{
name of optimizer (default: adam)
}
    \item{validation_split}{
proportion of validation data, it is ignored when there is a validation set (default: 0)
}
    \item{x}{
result of the function "fit_GAN"
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
trained GAN model
}
    \item{generator}{
trained generator model
}
    \item{discriminator}{
trained discriminator model
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
}


\author{
Dongmin Jung
}


\seealso{
keras::train_on_batch, keras::evaluate, keras::compile, CatEncoders::LabelEncoder.fit, CatEncoders::transform, CatEncoders::inverse.transform
}


\references{
Liebowitz, J. (Ed.). (2020). Data Analytics and AI. CRC Press.

Pedrycz, W., & Chen, S. M. (Eds.). (2020). Deep Learning: Concepts and Architectures. Springer.

Suguna, S. K., Dhivya, M., & Paiva, S. (Eds.). (2021). Artificial Intelligence (AI): Recent Trends and Applications. CRC Press.

Sun, S., Mao, L., Dong, Z., & Wu, L. (2019). Multiview machine learning. Springer.
}


\examples{
# model parameters
length_seq <- 403
embedding_dim <- 8
latent_dim <- 4
epochs <- 2
batch_size <- 64

# GAN
GAN_result <- fit_GAN(prot_seq = example_PTEN,
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
                    epochs = epochs,
                    batch_size = batch_size)
set.seed(1)
gen_prot_GAN <- gen_GAN(GAN_result, num_seq = 100)


### from preprocessing
GAN_result2 <- fit_GAN(preprocessing = GAN_result$preprocessing,
                        intermediate_generator_layers = list(
                            layer_dense(units = 16),
                            layer_dense(units = 128)),
                        intermediate_discriminator_layers = list(
                            layer_dense(units = 128, activation = "relu"),
                            layer_dense(units = 16, activation = "relu")),
                        epochs = epochs,
                        batch_size = batch_size)
gen_prot_GAN <- gen_GAN(GAN_result2, num_seq = 100)
}
