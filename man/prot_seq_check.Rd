\name{prot_seq_check}
\alias{prot_seq_check}


\title{
Check a protein sequence
}

\description{
The protein sequence dataset is filtered by eliminating sequences containing the non-amino acid characters (digits and blank spaces) from the amino acid sequences. A valid amino acid sequence means a string that only contains capital letters of an alphabet and a hyphen for a gap.
}

\usage{
prot_seq_check(prot_seq, label = NULL)
}

\arguments{
    \item{prot_seq}{
amino acid sequences
}
    \item{label}{
label (default: NULL)
}
}

\value{
valid sequences
}

\author{
Dongmin Jung
}

\references{
Mukhopadhyay, C. S., Choudhary, R. K., & Iquebal, M. A. (2017). Basic Applied Bioinformatics. John Wiley & Sons.
}

\examples{
prot_seq_check(example_PTEN[1])
}
