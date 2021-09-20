\name{ART}
\alias{fit_ART}
\alias{gen_ART}

\title{
Autoregressive language model with Transformer
}

\description{
The autoregressive generative model predicts the next amino acid in a protein given the amino acid sequence up to that point. The autoregressive model generates proteins one amino acid at a time. For one step of generation, it takes a context sequence of amino acids as input and outputs a probability distribution over amino acids. We sample from that distribution and then update the context sequence with the sampled amino acid. The Transformer is used as an encoder model. The AR with the Transformer model can be trained by the function "fit_ART", and then the function "gen_ART" generates protein sequences.
}

\usage{
fit_ART(prot_seq,
        length_seq,
        embedding_dim,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        layers = NULL,
        prot_seq_val = NULL,
        epochs,
        batch_size,
        preprocessing = list(
            x_train = NULL,
            x_val = NULL,
            y_train = NULL,
            y_val = NULL,
            lenc = NULL,
            length_seq = NULL,
            num_AA = NULL,
            embedding_dim = NULL,
            removed_prot_seq = NULL,
            removed_prot_seq_val = NULL),
        use_generator = FALSE,
        optimizer = "adam",
        metrics = "accuracy",
        validation_split = 0, ...) 

gen_ART(x,
        seed_prot,
        length_AA,
        method = NULL,
        b = NULL,
        t = 1,
        k = NULL,
        p = NULL)
}

\arguments{
    \item{prot_seq}{
amino acid sequence
}
    \item{length_seq}{
length of sequence used as input
}
    \item{embedding_dim}{
dimension of the dense embedding
}
    \item{num_heads}{
number of attention heads
}
    \item{ff_dim}{
hidden layer size in feedforward network inside transformer
}
    \item{num_transformer_blocks}{
number of transformer blocks
}
    \item{layers}{
list of layers between the transformer encoder and the output layer (default: NULL)
}
    \item{prot_seq_val}{
amino acid sequence for validation (default: NULL)
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
    \item num_AA : number of amino acids
    \item embedding_dim : dimension of the dense embedding
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
    \item{metrics}{
name of metrics (default: accuracy)
}
    \item{validation_split}{
proportion of validation data, it is ignored when there is a validation set (default: 0)
}
    \item{...}{
additional parameters for the "fit"
}
    \item{x}{
result of the function "fit_ART"
}
    \item{seed_prot}{
sequence to be used as seed proteins
}
    \item{length_AA}{
length of amino acids to be generated
}
    \item{method}{
"greedy", "beam", "temperature", "top_k", or "top_p"
}
    \item{b}{
beam size in the beam search
}
    \item{t}{
temperature in the temperature sampleing
}
    \item{k}{
number of amino acids in the top-k sampling
}
    \item{p}{
minimum probabilty for the set of amino acids in the top-p sampling
}
}


\value{
    \item{model}{
trained ART model
}
    \item{preprocessing}{
preprocessed results
}
}


\author{
Dongmin Jung
}


\references{
Deepak, P., Chakraborty, T., & Long, C. (2021). Data Science for Fake News: Surveys and Perspectives (Vol. 42). Springer.

Liu, Z., Lin, Y., & Sun, M. (2020). Representation learning for natural language processing. Springer.

Madani, A., McCann, B., Naik, N., Keskar, N. S., Anand, N., Eguchi, R. R., Huang, P., & Socher, R. (2020). Progen: Language modeling for protein generation. arXiv:2004.03497.
}


\seealso{
keras::fit, keras::compile, ttgsea::sampling_generator, DeepPINCS::multiple_sampling_generator, DeepPINCS::seq_preprocessing, DeepPINCS::get_seq_encode_pad, CatEncoders::LabelEncoder.fit, CatEncoders::transform, CatEncoders::inverse.transform
}


\examples{
prot_seq <- DeepPINCS::SARS_CoV2_3CL_Protease

# model parameters
length_seq <- 10
embedding_dim <- 16
num_heads <- 2
ff_dim <- 16
num_transformer_blocks <- 2
batch_size <- 32
epochs <- 2

# ART
ART_result <- fit_ART(prot_seq = prot_seq,
                    length_seq = length_seq,
                    embedding_dim = embedding_dim,
                    num_heads = num_heads,
                    ff_dim = ff_dim,
                    num_transformer_blocks = num_transformer_blocks,
                    layers = list(layer_dropout(rate = 0.1),
                                layer_dense(units = 32, activation = "relu"),
                                layer_dropout(rate = 0.1)),
                    prot_seq_val = prot_seq,
                    epochs = epochs,
                    batch_size = batch_size,
                    use_generator = TRUE,
                    callbacks = callback_early_stopping(
                        monitor = "val_loss",
                        patience = 10,
                        restore_best_weights = TRUE))

seed_prot <- "SGFRKMAFPS"
gen_ART(ART_result, seed_prot, length_AA = 20, method = "greedy")
gen_ART(ART_result, seed_prot, length_AA = 20, method = "beam", b = 5)
gen_ART(ART_result, seed_prot, length_AA = 20, method = "temperature", t = 0.1)
gen_ART(ART_result, seed_prot, length_AA = 20, method = "top_k", k = 3)
gen_ART(ART_result, seed_prot, length_AA = 20, method = "top_p", p = 0.75)


### from preprocessing
ART_result2 <- fit_ART(num_heads = 4,
                    ff_dim = 32,
                    num_transformer_blocks = 3,
                    layers = list(layer_dropout(rate=0.1),
                                layer_dense(units=32, activation="relu"),
                                layer_dropout(rate=0.1)),
                    epochs = epochs,
                    batch_size = batch_size,
                    preprocessing = ART_result$preprocessing,
                    use_generator = TRUE,
                    callbacks = callback_early_stopping(
                            monitor = "val_loss",
                            patience = 50,
                            restore_best_weights = TRUE))

gen_ART(ART_result2, seed_prot, length_AA = 20, method = "greedy")
gen_ART(ART_result2, seed_prot, length_AA = 20, method = "beam", b = 5)
gen_ART(ART_result2, seed_prot, length_AA = 20, method = "temperature", t = 0.1)
gen_ART(ART_result2, seed_prot, length_AA = 20, method = "top_k", k = 3)
gen_ART(ART_result2, seed_prot, length_AA = 20, method = "top_p", p = 0.75)
}
