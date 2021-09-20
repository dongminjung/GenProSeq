fit_GAN <- function(prot_seq,
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
                    validation_split = 0) {
    result <- NULL
    result$preprocessing <- NULL
    
    ### pre-processing
    if (all(unlist(lapply(preprocessing, Negate(is.null))))) {
        result$preprocessing <- preprocessing
        x_train <- preprocessing$x_train
        y_train <- preprocessing$y_train
        x_val <- preprocessing$x_val
        y_val <- preprocessing$y_val
        length_seq <- preprocessing$length_seq
        num_seq <- preprocessing$num_seq
        num_seq_val <- preprocessing$num_seq_val
        embedding_dim <- preprocessing$embedding_dim
        embedding_matrix <- preprocessing$embedding_matrix
        if (is.null(latent_dim)) latent_dim <- preprocessing$latent_dim
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
        num_seq <- dim(x_train)[1]
        y_train <- label
        embedding_matrix <- prot2vec_train$embedding_matrix
        
        x_val <- NULL
        y_val <- NULL
        num_seq_val <- NULL
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
            y_val <- label_val
            num_seq_val <- dim(x_val)[1]
        }
        
        if (is.null(prot_seq_val) & validation_split) {
            x <- x_train
            idx <- sample(seq_len(nrow(x)))
            train_idx <- seq_len(nrow(x)) %in%
                idx[seq_len(round(nrow(x) * (1 - validation_split)))]
            x_train <- x[train_idx,,]
            x_val <- x[!train_idx,,]
            if (!is.null(y_train) & is.null(y_val)) {
                y <- y_train
                y_train <- y[train_idx]
                y_val <- y[!train_idx]
            }
            num_seq <- dim(x_train)[1]
            num_seq_val <- dim(x_val)[1]
        }
    }
    
    
    ### building model
    # GAN
    g_model <- function(latent_dim, label) {
        generator_input <- layer_input(shape = c(latent_dim))
        if (!is.null(label)) {
            input_latent <- layer_input(shape = c(latent_dim))
            input_label <- layer_input(shape = c(1))
            embedding_label <- input_label %>%
                layer_embedding(input_dim = length(unique(label)), output_dim = latent_dim) %>%
                layer_flatten()
            generator_input <- layer_multiply(list(input_latent, embedding_label))
        }
        
        x <- generator_input
        for (i in seq_len(length(intermediate_generator_layers))) {
            x <- intermediate_generator_layers[[i]](x)
        }
        
        generator_output <- x %>%
            layer_dense(units = length_seq * embedding_dim) %>%
            layer_reshape(c(length_seq, embedding_dim))
        
        if (is.null(label)) {
            generator <- keras_model(generator_input, generator_output)
        } else {
            generator <- keras_model(list(input_latent, input_label), generator_output)
        }
    }
    
    d_model <- function(length_seq, embedding_dim, label, optimizer) {
        discriminator_input <- layer_input(shape = c(length_seq, embedding_dim))
        discriminator_output <- discriminator_input %>%
            layer_flatten(input_shape = c(length_seq, embedding_dim))
        
        x <- discriminator_output
        for (i in seq_len(length(intermediate_discriminator_layers))) {
            x <- intermediate_discriminator_layers[[i]](x)
        }
        
        if (is.null(label)) {
            discriminator_output <- x %>%
                layer_dense(units = 1, activation = "sigmoid")
            discriminator <- keras_model(discriminator_input, discriminator_output)
        } else {
            fake <- discriminator_output %>% 
                layer_dense(units = 1, activation = "sigmoid", name = "generation")
            aux <- discriminator_output %>%
                layer_dense(units = length(unique(label)), activation = "softmax", name = "auxiliary")
            discriminator <- keras_model(discriminator_input, list(fake, aux))
        }
        
        if (is.null(label)) {
            discriminator %>% compile(
                optimizer = optimizer,
                loss = "binary_crossentropy"
            )        
        } else {
            discriminator %>% compile(
                optimizer = optimizer,
                loss = list("binary_crossentropy", "sparse_categorical_crossentropy")
            )
        }
    }
    
    combined_model <- function(g_model, d_model, label, optimizer) {
        freeze_weights(d_model)
        
        if (is.null(label)) {
            gan <- keras_model_sequential() %>% g_model %>% d_model
        } else {
            input_latent <- layer_input(shape = c(latent_dim))
            input_label <- layer_input(shape = c(1))
            results <- g_model(list(input_latent, input_label)) %>% d_model
            gan <- keras_model(list(input_latent, input_label), results)
        }
        
        gan_optimizer <- optimizer_adam()
        
        if (is.null(label)) {
            gan %>% compile(
                optimizer = optimizer, 
                loss = list("binary_crossentropy")
            )
        } else {
            gan %>% compile(
                optimizer = optimizer,
                loss = list("binary_crossentropy", "sparse_categorical_crossentropy")
            )
        }
    }
    
    generator <- g_model(latent_dim, y_train)
    discriminator <- d_model(length_seq, embedding_dim, y_train, optimizer)
    combined <- combined_model(generator, discriminator, y_train, optimizer)
    
    
    ### training
    message("training...")
    if (!is.null(y_train)) {
        if (any(unlist(lapply(preprocessing, is.null)))) {
            lenc <- CatEncoders::LabelEncoder.fit(y_train)
            y_train <- CatEncoders::transform(lenc, y_train) - 1
            result$preprocessing$lenc <- lenc
            if (!is.null(y_val)) {
                y_val <- CatEncoders::transform(lenc, y_val) - 1
            }
        }
    }
    
    # epoch
    for (epoch in seq_len(epochs)) {
        epoch_gen_loss <- NULL
        epoch_disc_loss <- NULL
        
        num_batches <- trunc(num_seq/batch_size)
        possible_indexes <- seq_len(num_seq)
        
        cat(sprintf("Epoch %s/%s \n", epoch, epochs))
        progbar <- tensorflow::tf$keras$utils$Progbar(target = num_batches)
        
        # batch
        for (index in seq_len(num_batches)) {
            batch <- sample(possible_indexes, size = batch_size)
            possible_indexes <- possible_indexes[!possible_indexes %in% batch]
            
            
            # train discriminator
            noise <- matrix(stats::rnorm(batch_size * latent_dim), 
                            nrow = batch_size, ncol = latent_dim)
            x_train_batch <- x_train[batch,,,drop = FALSE]
            
            if (!is.null(y_train)) {
                y_train_batch <- y_train[batch]
                y_train_sample <- sample(unique(y_train), batch_size, replace = TRUE)
                aux_y <- matrix(c(y_train_sample, y_train_batch), ncol = 1)
                x_gen_batch <- predict(generator, list(noise, y_train_sample))
            } else {
                x_gen_batch <- predict(generator, noise)
            }
            
            X <- array(0, dim = c(2 * batch_size, length_seq, embedding_dim))
            X[seq(batch_size),,] <- x_gen_batch
            X[seq(batch_size + 1, 2 * batch_size),,] <- x_train_batch
            
            fake_real <- c(rep(0, batch_size), rep(1, batch_size))
            
            if (!is.null(y_train)) {
                disc_loss <- train_on_batch(
                    discriminator,
                    x = X,
                    y = list(fake_real, aux_y)
                )
            } else {
                disc_loss <- train_on_batch(
                    discriminator,
                    x = X,
                    y = fake_real
                )
            }
            epoch_disc_loss <- rbind(epoch_disc_loss, unlist(disc_loss))
            
            
            # train generator
            noise <- matrix(stats::rnorm(2 * batch_size * latent_dim), 
                            nrow = 2 * batch_size, ncol = latent_dim)
            
            if (!is.null(y_train)) {
                y_train_sample <- matrix(
                    sample(unique(y_train), size = 2 * batch_size, replace = TRUE),
                    ncol = 1)
            }
            
            misleading_real <- rep(1, 2 * batch_size)
            
            if (!is.null(y_train)) {
                combined_loss <- train_on_batch(
                    combined,
                    list(noise, y_train_sample),
                    list(misleading_real, y_train_sample)
                )
            } else {
                combined_loss <- train_on_batch(
                    combined,
                    noise,
                    misleading_real)
            }
            epoch_gen_loss <- rbind(epoch_gen_loss, unlist(combined_loss))
            progbar$add(1)
        }
        
        discriminator_train_loss <- apply(epoch_disc_loss, 2, mean)
        generator_train_loss <- apply(epoch_gen_loss, 2, mean)
        
        if (!is.null(x_val)) {
            
            # evaluate discriminator
            noise <- matrix(stats::rnorm(num_seq_val * latent_dim), 
                            nrow = num_seq_val, ncol = latent_dim)
            
            if (!is.null(y_val)) {
                y_val_sample <- sample(unique(y_val), size = num_seq_val, replace = TRUE)
                aux_y <- matrix(c(y_val_sample, y_val), ncol = 1)
                x_gen <- predict(generator, list(noise, y_val_sample))
            } else {
                x_gen <- predict(generator, noise)
            }
            
            X <- array(0, dim = c(2 * num_seq_val, length_seq, embedding_dim))
            X[seq(num_seq_val),,] <- x_gen
            X[seq(num_seq_val + 1, 2 * num_seq_val),,] <- x_val
            
            fake_real <- c(rep(0, num_seq_val), rep(1, num_seq_val))
            
            if (!is.null(y_val)) {
                discriminator_test_loss <- evaluate(
                    discriminator,
                    x = X,
                    y = list(fake_real, aux_y),
                    verbose = FALSE
                ) %>% unlist()
            } else {
                discriminator_test_loss <- evaluate(
                    discriminator,
                    x = X,
                    y = fake_real,
                    verbose = FALSE
                ) %>% unlist()
            }
            
            
            # evaluate generator
            noise <- matrix(stats::rnorm(2 * num_seq_val * latent_dim), 
                            nrow = 2 * num_seq_val, ncol = latent_dim)
            
            if (!is.null(y_val)) {
                y_val_sample <- matrix(
                    sample(unique(y_val), size = 2 * num_seq_val, replace = TRUE),
                    ncol = 1)
            }
            
            misleading_real <- rep(1, 2 * num_seq_val)
            
            if (!is.null(y_val)) {
                generator_test_loss <- combined %>% evaluate(
                    list(noise, y_val_sample),
                    list(misleading_real, y_val_sample),
                    verbose = FALSE
                )
            } else {
                generator_test_loss <- combined %>% evaluate(
                    noise,
                    misleading_real,
                    verbose = FALSE
                )
            }
        }
        
        
        # performance
        if (length(discriminator$metrics_names) == 1) {
            row_fmt <- "%s : loss %f \n"
        } else {
            row_fmt <- "%s : loss %f | generation_loss %f | auxiliary_loss %f \n"
        }
        
        cat(do.call(sprintf, c(list(row_fmt, "generator (train)"), generator_train_loss)))
        if (!is.null(x_val)) {
            cat(do.call(sprintf, c(list(row_fmt, "generator (test)"), generator_test_loss)))
        }
        cat(do.call(sprintf, c(list(row_fmt, "discriminator (train)"), discriminator_train_loss)))
        if (!is.null(x_val)) {
            cat(do.call(sprintf, c(list(row_fmt, "discriminator (test)"), discriminator_test_loss)))
        }
        cat("\n")
    }
    
    result$model <- combined
    result$generator <- generator
    result$discriminator <- discriminator
    result$preprocessing$x_train <- x_train
    result$preprocessing$y_train <- y_train
    result$preprocessing$x_val <- x_val
    result$preprocessing$y_val <- y_val
    result$preprocessing$length_seq <- length_seq
    result$preprocessing$num_seq <- num_seq
    result$preprocessing$num_seq_val <- num_seq_val
    result$preprocessing$embedding_dim <- embedding_dim
    result$preprocessing$embedding_matrix <- embedding_matrix
    result$preprocessing$latent_dim <- latent_dim
    result
}





gen_GAN <- function(x, label = NULL, num_seq, remove_gap = TRUE) {
    result <- NULL
    latent_dim <- x$preprocessing$latent_dim
    
    ### generating
    message("generating...")
    if (is.null(x$preprocessing$y_train)) {
        gen_vec <- predict(x$generator,
                        matrix(stats::rnorm(num_seq * latent_dim),
                                nrow = num_seq, ncol = latent_dim))
    } else {
        lenc <- x$preprocessing$lenc
        lable_enc <- CatEncoders::transform(lenc, label) - 1
        gen_vec <- predict(x$generator,
                        list(matrix(stats::rnorm(num_seq * latent_dim),
                                    nrow = num_seq, ncol = latent_dim),
                                    lable_enc))
    }
    
    
    ### post-processing
    message("post-processing...")
    # vector to sequence
    gen_seq <- vec2prot(gen_vec, x$preprocessing$embedding_matrix)
    
    if (remove_gap) {
        gen_seq <- gsub("-", "", gen_seq)
    } else {
        gen_seq <- gen_seq
    }
    
    result$gen_seq <- gen_seq
    result$label <- label
    result
}
