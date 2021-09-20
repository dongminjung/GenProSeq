\name{transformer}
\alias{layer_embedding_token_position}
\alias{layer_transformer_encoder}

\title{
Transformer model
}

\description{
The Transformer architecture is a nonrecurrent architecture with a series of attention-based blocks. Each block is composed of a multi-head attention layer and a position-wise feedforward layer with an add and normalize layer in between. These layers process input sequences simultaneously, in parallel, independently of sequential order.
}

\usage{
layer_embedding_token_position(x, maxlen, vocab_size, embed_dim)
layer_transformer_encoder(x, embed_dim, num_heads, ff_dim, num_transformer_blocks)
}

\arguments{
    \item{x}{
layer object
}
    \item{maxlen}{
maximum of sequence size
}
    \item{vocab_size}{
vacabulary size
}
    \item{embed_dim}{
embedding size for each token
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
}


\value{
layer object
}


\author{
Dongmin Jung
}


\references{
Lappin, S. (2021). Deep learning and linguistic representation. CRC Press.

Liu, Z., Lin, Y., & Sun, M. (2020). Representation learning for natural language processing. Springer.
}


\examples{
num_AA <- 20
length_seq <- 10
embedding_dim <- 16
num_heads <- 2
ff_dim <- 16
num_transformer_blocks <- 2

inputs <- layer_input(shape = length_seq)
x <- inputs \%>\%
    layer_embedding_token_position(maxlen = length_seq,
                                vocab_size = num_AA,
                                embed_dim = embedding_dim) \%>\%
    layer_transformer_encoder(embed_dim = embedding_dim,
                            num_heads = num_heads,
                            ff_dim = ff_dim,
                            num_transformer_blocks = num_transformer_blocks) \%>\%
    layer_global_average_pooling_1d()
}
