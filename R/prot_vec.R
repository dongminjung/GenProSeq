prot2vec <- function(prot_seq, embedding_dim, embedding_matrix = NULL, ...) {
    result <- NULL
    num_seq <- length(prot_seq)
    AA_seq <- strsplit(prot_seq, "")
    
    if(is.null(embedding_matrix)) {
        model <- word2vec::word2vec(x = gsub("", " ", prot_seq), dim = embedding_dim, min_count = 1, split = " ", ...)
        embedding_matrix <- as.matrix(model)
    } else {
        embedding_matrix <- embedding_matrix
    }
    
    length_seq <- nchar(prot_seq[1])
    prot_vec <- array(0, dim = c(num_seq, length_seq, ncol(embedding_matrix)))
    
    for (i in seq_len(num_seq)) {
        prot_vec[i,,] <- embedding_matrix[AA_seq[[i]],]
    }
    
    result$prot_vec <- prot_vec
    result$embedding_matrix <- embedding_matrix
    result
}





vec2prot <- function(prot_vec, embedding_matrix) {
    prot_seq <- NULL
    num_seq <- dim(prot_vec)[1]
    length_seq <- dim(prot_vec)[2]
    for (i in seq_len(num_seq)) {
        temp_prot_seq <- 0
        for (j in seq_len(length_seq)) {
            similarity <- word2vec::word2vec_similarity(prot_vec[i,j,], embedding_matrix)
            sorted_similarity <- gsub("[^A-Z|-]", "", colnames(similarity)[order(similarity, decreasing = TRUE)])
            temp_prot_seq[j] <- sorted_similarity[sorted_similarity != ""][1]
        }
        prot_seq  <- c(prot_seq, paste(temp_prot_seq, collapse = ""))
    }
    prot_seq
}
