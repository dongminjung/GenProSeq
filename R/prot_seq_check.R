prot_seq_check <- function(prot_seq, label = NULL) {
    result <- NULL
    
    if (!is.null(label)) {
        if (length(prot_seq) != length(label)) {
            stop("check the number of samples")
        }
    }
    
    check <- NULL
    for (i in seq_len(length(prot_seq))) {
        check[i] <- grepl("^[A-Z-]+$", prot_seq[i])
    }
    
    if (!all(check)) {
        result$removed_prot_seq <- which(!check)
        message("at least one of protein sequences may not be valid")
    }
    
    result$prot_seq <- prot_seq[check]
    result$label <- label[check]
    result
}
