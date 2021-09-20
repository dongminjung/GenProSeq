fit_ART <- function(prot_seq,
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
                    validation_split = 0, ...) {
    result <- NULL
    result$preprocessing <- NULL
    x_train <- NULL
    x_val <- NULL
    y_train <- NULL
    y_val <- NULL
    
    ### pre-processing
    if (all(unlist(lapply(preprocessing, Negate(is.null))))) {
        result$preprocessing <- preprocessing
        x_train <- preprocessing$x_train
        x_val <- preprocessing$x_val
        y_train <- preprocessing$y_train
        y_val <- preprocessing$y_val
        length_seq <- preprocessing$length_seq
        embedding_dim <- preprocessing$embedding_dim
        lenc <- preprocessing$lenc
        num_AA <- preprocessing$num_AA
        if (embedding_dim %% num_heads != 0) {
            stop("the embedding dimension is a multiple of the number of attention heads")
        }
    } else {
        if (any(unlist(lapply(preprocessing, is.null)))) {
            message("pre-processing...")
            
            if (embedding_dim %% num_heads != 0) {
                stop("the embedding dimension is a multiple of the number of attention heads")
            }
            checked_prot_seq <- prot_seq_check(prot_seq = prot_seq)
            prot_seq <- checked_prot_seq$prot_seq
            result$preprocessing$removed_prot_seq <- checked_prot_seq$removed_prot_seq
            
            temp <- NULL
            seq_encode_pad <- DeepPINCS::get_seq_encode_pad(prot_seq, max(nchar(prot_seq)))
            lenc <- seq_encode_pad$lenc
            num_AA <- seq_encode_pad$num_tokens
            for (i in seq_len(length(prot_seq))) {
                temp <- rbind(temp, embed(rev(seq_encode_pad$sequences_encode_pad[i,][seq_encode_pad$sequences_encode_pad[i,] != 0]),
                                        length_seq+1))
            }
            x_train <- temp[, seq_len(length_seq)]
            y_train <- temp[, length_seq+1] - 1
            
            if (!is.null(prot_seq_val)) {
                checked_prot_seq_val <- prot_seq_check(prot_seq = prot_seq_val)
                prot_seq_val <- checked_prot_seq_val$prot_seq
                result$preprocessing$removed_prot_seq_val <- checked_prot_seq_val$removed_prot_seq_val
                
                seq_val_encode_pad <- DeepPINCS::seq_preprocessing(AAseq = as.matrix(prot_seq_val),
                                                                    type = "sequence",
                                                                    length_seq = max(nchar(prot_seq_val)),
                                                                    lenc = lenc)
                temp_val <- NULL
                for (i in seq_len(length(prot_seq_val))) {
                    temp_val <- rbind(temp_val,
                                    embed(rev(seq_val_encode_pad$sequences_encode_pad[[1]][i,][seq_val_encode_pad$sequences_encode_pad[[1]][i,] != 0]),
                                        length_seq + 1))
                }
                x_val <- temp_val[, seq_len(length_seq)]
                y_val <- temp_val[, length_seq+1] - 1
            }
            
            if (is.null(prot_seq_val) & validation_split) {
                temp_x_train <- x_train
                temp_y_train <- y_train
                idx <- sample(seq_len(nrow(temp_x_train)))
                train_idx <- seq_len(nrow(temp_x_train)) %in%
                    idx[seq_len(round(nrow(temp_x_train) * (1 - validation_split)))]
                x_train <- temp_x_train[train_idx,]
                x_val <- temp_x_train[!train_idx,]
                y_train <- temp_y_train[train_idx]
                y_val <- temp_y_train[!train_idx]
            }
        }
    }
    
    
    # model
    inputs <- layer_input(shape = length_seq)
    x <- inputs %>%
        layer_embedding_token_position(maxlen = length_seq,
                                        vocab_size = num_AA,
                                        embed_dim = embedding_dim) %>%
        layer_transformer_encoder(embed_dim = embedding_dim,
                                num_heads = num_heads,
                                ff_dim = ff_dim,
                                num_transformer_blocks = num_transformer_blocks) %>%
        layer_global_average_pooling_1d()
    
    if (!is.null(layers)) {
        for (i in seq_len(length(layers))) {
            x <- layers[[i]](x)
        }
    }
    
    outputs <- layer_dense(units = num_AA, activation = "softmax")(x)
    model <- keras_model(inputs = inputs, outputs = outputs)
    model %>% compile(optimizer = optimizer,
                    loss = "sparse_categorical_crossentropy",
                    metrics = metrics)
    
    
    ### training
    message("training...")
    if (!use_generator) {
        # without generator
        validation_data <- NULL
        if (all(!is.null(x_val), !is.null(y_val))) {
            validation_data <- list(x_val, y_val)
        }
        
        model %>% keras::fit(x_train, y_train, epochs = epochs, batch_size = batch_size,
                            validation_data = validation_data,
                            validation_split = validation_split, ...)
    } else {
        # with generator
        validation_data <- NULL
        validation_steps <- NULL
        if (all(!is.null(x_val), !is.null(y_val))) {
            validation_data <- ttgsea::sampling_generator(
                as.matrix(x_val), as.matrix(y_val),
                batch_size = batch_size)
            validation_steps <- ceiling(nrow(x_val)/batch_size)
        }
        
        model %>% keras::fit(ttgsea::sampling_generator(as.matrix(x_train), 
                                                        as.matrix(y_train),
                                                        batch_size = batch_size),
                                steps_per_epoch = nrow(x_train)/batch_size,
                                epochs = epochs,
                                validation_data = validation_data,
                                validation_steps = validation_steps,...)
    }
    
    
    result$preprocessing$x_train <- x_train
    result$preprocessing$y_train <- y_train
    result$preprocessing$x_val <- x_val
    result$preprocessing$y_val <- y_val
    result$preprocessing$length_seq <- length_seq
    result$preprocessing$embedding_dim <- embedding_dim
    result$preprocessing$lenc <- lenc
    result$preprocessing$num_AA <- num_AA
    result$model <- model
    result
}





gen_ART <- function(x, seed_prot, length_AA, method = NULL,
                    b = NULL, t = 1, k = NULL, p = NULL) {
    model <- x$model
    length_seq <- x$preprocessing$length_seq
    lenc <- x$preprocessing$lenc
    labels <- sort(unique(x$preprocessing$y_train))
    
    
    ### generating
    message("generating...")
    temp_prot = prot <- seed_prot
    X <- DeepPINCS::get_seq_encode_pad(temp_prot, length_seq = length_seq, lenc = lenc)$sequences_encode_pad
    for (i in seq_len(length_AA)) {
        pred <- predict(model, rbind(X))
        if (method == "greedy") {
            temp_X <- apply(pred, 1, which.max)
        } else if (method == "beam") {
            temp_X <- apply(pred, 1,
                            function(x) {
                                candidate <- labels[order(x, decreasing = TRUE)[seq_len(b)]] + 1
                                candidate_prob <- x[order(x, decreasing = TRUE)[seq_len(b)]]
                                candidate_next_prob <- 0
                                for (j in seq_len(b)) {
                                    candidate_next_prob[j] <- max(predict(model, rbind(c(X[-1], candidate[j]))))
                                }
                                candidate[which.max(candidate_prob * candidate_next_prob)]
                            })
        } else if (method == "temperature") {
            temp_X <- apply(pred, 1,
                            function(x) {
                                pred_temp <- log(x)/t
                                pred_temp <- exp(pred_temp)/sum(exp(pred_temp))
                                sample(length(x), size = 1, prob = pred_temp)
                            })
            
        } else if (method == "top_k") {
            temp_X <- apply(pred, 1,
                            function(x) sample(order(x, decreasing = TRUE)[seq_len(k)], size = 1,
                                            prob = x[order(x, decreasing = TRUE)[seq_len(k)]]/
                                                sum(x[order(x, decreasing = TRUE)[seq_len(k)]])))
        } else if (method == "top_p") {
            temp_X <- apply(pred[1,,drop=FALSE], 1,
                            function(x) {
                                k <- min(which((cumsum(x[order(x, decreasing = TRUE)]) > p) == TRUE))
                                if (length(order(x, decreasing = TRUE)[seq_len(k)]) == 1) {
                                    order(x, decreasing = TRUE)[seq_len(k)]
                                } else {
                                    sample(order(x, decreasing = TRUE)[seq_len(k)], size = 1,
                                        prob = x[order(x, decreasing = TRUE)[seq_len(k)]]/
                                            sum(x[order(x, decreasing = TRUE)[seq_len(k)]]))
                                }
                            })
        } else {
            stop("method is not available")
        }
        
        temp_X <- labels[temp_X] + 1
        gen_AA <- CatEncoders::inverse.transform(lenc, temp_X)
        prot <- paste(prot, gen_AA, sep = "")
        temp_prot <- substr(prot, nchar(gen_AA)-length_seq+1, nchar(gen_AA))
        X <- c(X[-1], temp_X)
    }
    prot
}
