\name{prot_vec}
\alias{prot2vec}
\alias{vec2prot}

\title{
Converting from protein sequences to vectors or vice versa.
}

\description{
By using the word2vec model, amino acids are mapped to vectors of real numbers. Conceptually, it involves a mathematical embedding from a space with many dimensions per amino acid to a continuous vector space with a much lower dimension.
}

\usage{
prot2vec(prot_seq, embedding_dim, embedding_matrix = NULL, ...)
vec2prot(prot_vec, embedding_matrix)
}

\arguments{
    \item{prot_seq}{
protein sequences
}
    \item{prot_vec}{
protein embedding vectors
}
    \item{embedding_dim}{
dimension of embedding vectors
}
    \item{embedding_matrix}{
embedding matrix (default: NULL)
}
    \item{...}{
arguments for "word2vec::word2vec" but for dim, min_count and split 
}
}


\value{
    \item{prot_seq}{
protein sequences
}
    \item{prot_vec}{
protein embedding vectors
}
    \item{embedding_matrix}{
embedding matrix
}
}


\author{
Dongmin Jung
}


\seealso{
word2vec::word2vec, word2vec::word2vec_similarity
}


\references{
Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. arXiv:1310.4546.

Chang, M. (2020). Artificial intelligence for drug development, precision medicine, and healthcare.
}


\examples{
prot_seq <- example_PTEN[1:10]
prot2vec_result <- prot2vec(prot_seq = prot_seq, embedding_dim = 8)
vec2prot_result <- vec2prot(prot_vec = prot2vec_result$prot_vec,
                            embedding_matrix = prot2vec_result$embedding_matrix)
}
