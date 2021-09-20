fit_VAE <- function(prot_seq,
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
                    validation_split = 0, ...) {
    if (tensorflow::tf$executing_eagerly())
        tensorflow::tf$compat$v1$disable_eager_execution()
    
    result <- NULL
    result$preprocessing <- NULL
    if (regularization < 0) {
        stop("regularization parameter should be nonnegative")
    }
    
    
    ### pre-processing
    if (all(unlist(lapply(preprocessing, Negate(is.null))))) {
        result$preprocessing <- preprocessing
        x_train <- preprocessing$x_train
        y_train <- preprocessing$y_train
        x_val <- preprocessing$x_val
        y_val <- preprocessing$y_val
        length_seq <- preprocessing$length_seq
        num_seq <- preprocessing$num_seq
        embedding_dim <- preprocessing$embedding_dim
        embedding_matrix <- preprocessing$embedding_matrix
    }
    
    if (any(unlist(lapply(preprocessing, is.null)))) {
        message("pre-processing...")
        
        # check
        checked_prot_seq <- prot_seq_check(prot_seq = prot_seq, label = label)
        prot_seq <- checked_prot_seq$prot_seq
        label <- checked_prot_seq$label
        result$preprocessing$removed_prot_seq <- checked_prot_seq$removed_prot_seq
        
        # sequence to vector
        prot2vec_train <- do.call(prot2vec,
                                c(list(prot_seq = prot_seq, embedding_dim = embedding_dim),
                                    embedding_args))
        x_train <- prot2vec_train$prot_vec
        num_seq <- length(prot_seq)
        x_train <- reticulate::array_reshape(x_train, c(num_seq, length_seq * embedding_dim))
        y_train <- label
        embedding_matrix <- prot2vec_train$embedding_matrix
        
        x_val <- NULL
        y_val <- NULL
        if (!is.null(prot_seq_val)) {
            # check
            checked_prot_seq_val <- prot_seq_check(prot_seq = prot_seq_val, label = label_val)
            prot_seq_val <- checked_prot_seq_val$prot_seq
            label_val <- checked_prot_seq_val$label
            result$preprocessing$removed_prot_seq_val <- checked_prot_seq_val$removed_prot_seq
            # sequence to vector
            prot2vec_val <- prot2vec(prot_seq = prot_seq_val,
                                    embedding_matrix = embedding_matrix)
            x_val <- prot2vec_val$prot_vec
            x_val <- reticulate::array_reshape(x_val, c(num_seq, length_seq * embedding_dim))
            y_val <- label_val
        }
        
        if (is.null(prot_seq_val) & validation_split) {
            x <- x_train
            idx <- sample(seq_len(nrow(x)))
            train_idx <- seq_len(nrow(x)) %in%
                idx[seq_len(round(nrow(x) * (1 - validation_split)))]
            x_train <- x[train_idx,]
            x_val <- x[!train_idx,]
            if (!is.null(y_train) & is.null(y_val)) {
                y <- y_train
                y_train <- y[train_idx]
                y_val <- y[!train_idx]
            }
        }
    }
    
    
    ### building model
    # VAE
    x <- layer_input(shape = c(length_seq * embedding_dim))
    if (is.null(y_train)) {
        encoded <- x
    } else {
        condition <- layer_input(shape = c(1), name = "condition")
        encoded <- layer_concatenate(c(x, condition))
    }
    
    for (i in seq_len(length(intermediate_encoder_layers))) {
        encoded <- intermediate_encoder_layers[[i]](encoded)
    }
    
    assign("z_mean", layer_dense(encoded, latent_dim, name = "z_mean"),
        envir = globalenv())
    assign("z_log_stddev", layer_dense(encoded, latent_dim, name = "z_log_stddev"),
        envir = globalenv())
    
    z <- layer_lambda(list(z_mean, z_log_stddev),
                    function(arg) {
                        z_mean <- arg[[1]]
                        z_log_stddev <- arg[[2]]
                        epsilon <- k_random_normal(shape = c(k_shape(z_mean)[[1]]),
                                                    mean = 0.0, stddev = 1)
                        z_mean + k_exp(z_log_stddev)*epsilon
                    }, name = "latent")
    
    if (is.null(y_train)) {
        decoded <- z
    } else {
        decoded <- layer_concatenate(c(z, condition))
    }
    decoder_layers <- c(intermediate_decoder_layers,
                        layer_dense(units = length_seq * embedding_dim))
    for (i in seq_len(length(decoder_layers))) {
        decoded <- decoder_layers[[i]](decoded)
    }
    
    if (is.null(y_train)) {
        model <- keras_model(inputs = x, outputs = decoded)
    } else {
        model <- keras_model(inputs = c(x, condition), outputs = decoded)
    }
    
    # encoder
    if (is.null(y_train)) {
        encoder <- keras_model(inputs = x, outputs = z_mean)
    } else {
        encoder <- keras_model(inputs = c(x, condition), outputs = z_mean)
    }
    
    # decoder
    if (is.null(y_train)) {
        decoder_input <- layer_input(shape = latent_dim)
    } else {
        decoder_input <- layer_input(shape = latent_dim + 1)
    }
    decoder_output <- decoder_input
    for (i in seq_len(length(decoder_layers))) {
        decoder_output <- decoder_layers[[i]](decoder_output)
    }
    decoder <- keras_model(decoder_input, decoder_output)
    
    # loss
    vae_loss <- function(x_true, x_pred) {
        xent_loss <- (length_seq * embedding_dim / 1.0) * loss_mean_squared_error(x_true, x_pred)
        kl_loss <- -0.5 * k_mean(1 + z_log_stddev - k_square(z_mean) - k_exp(z_log_stddev), axis = -1L)
        xent_loss + regularization * kl_loss
    }
    
    model %>% keras::compile(optimizer = optimizer, loss = vae_loss)
    
    
    ### training
    message("training...")
    if (is.null(y_train)) {
        # vanilla vae
        if (!use_generator) {
            # without generator
            validation_data <- NULL
            if (!is.null(x_val)) {
                validation_data <- list(x_val, x_val)
            }
            
            model %>% keras::fit(
                x_train, x_train,
                epochs = epochs, 
                batch_size = batch_size, 
                validation_data = validation_data,
                validation_split = validation_split, ...)
        } else {
            # with generator
            validation_data <- NULL
            validation_steps <- NULL
            if (!is.null(x_val)) {
                validation_data <- DeepPINCS::multiple_sampling_generator(
                    list(x_val), x_val,
                    batch_size = batch_size)
                validation_steps <- ceiling(nrow(x_val)/batch_size)
            }
            
            model %>% keras::fit(
                DeepPINCS::multiple_sampling_generator(
                    list(x_train), x_train,
                    batch_size = batch_size),
                steps_per_epoch = ceiling(nrow(x_train)/batch_size),
                epochs = epochs,
                validation_data = validation_data,
                validation_steps = validation_steps, ...)
        }
    } else {
        # conditional vae
        if (any(unlist(lapply(preprocessing, is.null)))) {
            lenc <- CatEncoders::LabelEncoder.fit(y_train)
            y_train <- CatEncoders::transform(lenc, y_train)
            result$preprocessing$lenc <- lenc
            
            if (!is.null(y_val)) {
                y_val <- CatEncoders::transform(lenc, y_val)
            }
        }
        
        if (!use_generator) {
            # without generator
            validation_data <- NULL
            if (!is.null(x_val)) {
                validation_data <- list(list(x_val, y_val), x_val)
            }
            
            model %>% keras::fit(
                list(x_train, y_train), x_train,
                shuffle = TRUE, 
                epochs = epochs, 
                batch_size = batch_size, 
                validation_data = validation_data,
                validation_split = validation_split, ...)
        } else {
            # with generator
            validation_data <- NULL
            validation_steps <- NULL
            if (!is.null(x_val)) {
                validation_data <- DeepPINCS::multiple_sampling_generator(
                    list(x_val, cbind(y_val)), cbind(x_val),
                    batch_size = batch_size)
                validation_steps <- ceiling(nrow(x_val)/batch_size)
            }
            
            model %>% keras::fit(
                DeepPINCS::multiple_sampling_generator(
                    list(x_train, cbind(y_train)), cbind(x_train),
                    batch_size = batch_size),
                steps_per_epoch = ceiling(nrow(x_train)/batch_size),
                epochs = epochs,
                validation_data = validation_data,
                validation_steps = validation_steps, ...)
        }
    }
    
    rm(z_mean, envir = globalenv())
    rm(z_log_stddev, envir = globalenv())
    
    result$model <- model
    result$encoder <- encoder
    result$decoder <- decoder
    result$preprocessing$x_train <- x_train
    result$preprocessing$y_train <- y_train
    result$preprocessing$x_val <- x_val
    result$preprocessing$y_val <- y_val
    result$preprocessing$length_seq <- length_seq
    result$preprocessing$embedding_dim <- embedding_dim
    result$preprocessing$embedding_matrix <- embedding_matrix
    result
}





gen_VAE <- function(x, label = NULL, num_seq, remove_gap = TRUE,
                    batch_size, use_generator = FALSE) {
    result <- NULL
    encoder <- x$encoder
    decoder <- x$decoder
    x_train <- x$preprocessing$x_train
    y_train <- x$preprocessing$y_train
    length_seq <- x$preprocessing$length_seq
    embedding_dim <- x$preprocessing$embedding_dim
    
    x_train_encoded <- predict(encoder, list(x_train, y_train))
    rownames(x_train_encoded) <- rownames(x_train)
    colnames(x_train_encoded) <- paste("latent", seq_len(ncol(x_train_encoded)))
    
    
    ### generating
    message("generating...")
    if (is.null(y_train)) {
        # vanilla vae
        label <- NULL
        model_BIC <- mclust::mclustBIC(x_train_encoded, verbose = FALSE)
        mod <- mclust::mclustModel(x_train_encoded, model_BIC)
        z_sample <- mclust::sim(modelName = mod$modelName, 
                                parameters = mod$parameters, 
                                n = num_seq)
        
        if (!use_generator) {
            x_gen <- predict(decoder, z_sample[,-1])
        } else {
            x_gen <- predict(decoder,
                            DeepPINCS::multiple_sampling_generator(
                                list(z_sample[,-1]),
                                batch_size = batch_size,
                                shuffle = FALSE),
                            steps = ceiling(nrow(z_sample)/batch_size))
        }
        
    } else {
        # conditional vae
        lenc <- x$preprocessing$lenc
        label_enc <- CatEncoders::transform(lenc, label)
        all_labels <- names(table(label_enc))
        z_sample <- matrix(0, num_seq, ncol(x_train_encoded))
        for (i in seq_len(length(all_labels))) {
            model_BIC <- mclust::mclustBIC(x_train_encoded[y_train == all_labels[i],],
                                        verbose = FALSE)
            mod <- mclust::mclustModel(x_train_encoded[y_train == all_labels[i],], model_BIC)
            temp_z <- mclust::sim(modelName = mod$modelName, 
                                parameters = mod$parameters, 
                                n = table(label_enc)[i])
            z_sample[label_enc == all_labels[i],] <- temp_z[,-1]
        }
        
        if (!use_generator) {
            gen_vec <- predict(decoder, cbind(z_sample, label_enc))
        } else {
            gen_vec <- predict(decoder,
                                DeepPINCS::multiple_sampling_generator(
                                    list(cbind(z_sample, label_enc)),
                                    batch_size = batch_size,
                                    shuffle = FALSE),
                                steps = ceiling(nrow(z_sample)/batch_size))
        }
    }
    
    
    ### post-processing
    message("post-processing...")
    # vector to sequence
    gen_vec <- reticulate::array_reshape(as.matrix(gen_vec),
                                        c(num_seq, length_seq, embedding_dim))
    gen_seq <- vec2prot(gen_vec, x$preprocessing$embedding_matrix)
    
    if(remove_gap) {
        gen_seq <- gsub("-", "", gen_seq)
    } else {
        gen_seq <- gen_seq
    }
    
    result$gen_seq <- gen_seq
    result$label <- label
    result$latent_vector <- x_train_encoded
    result
}
