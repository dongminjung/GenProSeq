layer_embedding_token_position <- function(x, maxlen, vocab_size, embed_dim) {
    layer_token_position_embedding <- Layer(
        classname = "TokenPositionEmbedding", 
        initialize = function(maxlen, vocab_size, embed_dim, ...) {
            super()$`__init__`(...)
            self$token_emb <- layer_embedding(
                input_dim = vocab_size + 1,
                output_dim = embed_dim,
                mask_zero = TRUE)
            self$pos_emb <- layer_embedding(input_dim = maxlen, output_dim = embed_dim)
        },
        call = function(inputs, ...) {
            maxlen <- tensorflow::tf$shape(inputs)[length(tensorflow::tf$shape(inputs))]
            positions <- tensorflow::tf$range(start = 0, limit = maxlen, delta = 1)
            positions <- self$pos_emb(positions)
            x <- self$token_emb(inputs)
            x + positions
        },
        get_config = function() {
            list(
                name = self$name
            )
        }
    )
    
    if (!is.null(globalenv()$maxlen)) old_maxlen <- maxlen
    if (!is.null(globalenv()$vocab_size)) old_vocab_size <- vocab_size
    if (!is.null(globalenv()$embed_dim)) old_embed_dim <- embed_dim
    
    assign("maxlen", maxlen, envir = globalenv())
    assign("vocab_size", vocab_size, envir = globalenv())
    assign("embed_dim", embed_dim, envir = globalenv())
    
    x <- layer_token_position_embedding(x,
                                        maxlen = maxlen,
                                        vocab_size = vocab_size,
                                        embed_dim = embed_dim)
    if (!exists("old_maxlen")) rm(maxlen, envir = globalenv())
    if (!exists("old_vocab_size")) rm(vocab_size, envir = globalenv())
    if (!exists("old_embed_dim")) rm(embed_dim, envir = globalenv())
    x
}





layer_transformer_encoder <- function(x, embed_dim, num_heads, ff_dim, num_transformer_blocks) {
    layer_MHA <- Layer(
        classname = "MultiHeadAttention", 
        initialize = function(embed_dim, num_heads, ...) {
            super()$`__init__`(...)
            self$embed_dim <- embed_dim
            self$num_heads <- num_heads
            
            stopifnot(embed_dim %% self$num_heads == 0)
            
            self$depth <- embed_dim %/% num_heads
            self$query_dense <- layer_dense(units = embed_dim)
            self$key_dense <- layer_dense(units = embed_dim)
            self$value_dense <- layer_dense(units = embed_dim)
            self$dense <- layer_dense(units = embed_dim)
        },
        scaled_dot_product_attention = function(query, key, value) {
            matmul_qk <- tensorflow::tf$matmul(query, key, transpose_b = TRUE)
            dk <- tensorflow::tf$cast(tensorflow::tf$shape(key)[length(tensorflow::tf$shape(key))],
                                    tensorflow::tf$float32)
            scaled_attention_logits <- matmul_qk / tensorflow::tf$math$sqrt(dk)
            attention_weights <- tensorflow::tf$nn$softmax(scaled_attention_logits, axis = -1)
            output <- tensorflow::tf$matmul(attention_weights, value)
            output
        },
        split_heads = function(x, batch_size) {
            x <- tensorflow::tf$reshape(x, list(batch_size, -1L,
                                                as.integer(self$num_heads),
                                                as.integer(self$depth)))
            tensorflow::tf$transpose(x, perm=c(0L, 2L, 1L, 3L))
        },
        call = function(inputs, ...) {
            batch_size <- tensorflow::tf$shape(inputs)[1]
            
            query <- self$query_dense(inputs)
            key <- self$key_dense(inputs)
            value <- self$value_dense(inputs)
            
            query <- self$split_heads(query, batch_size)  
            key <- self$split_heads(key, batch_size)
            value <- self$split_heads(value, batch_size)
            
            scaled_attention <- self$scaled_dot_product_attention(query, key, value)
            scaled_attention <- tensorflow::tf$transpose(scaled_attention, perm=c(0L, 2L, 1L, 3L))  
            
            concat_attention <- tensorflow::tf$reshape(scaled_attention,
                                                    list(batch_size, -1L,
                                                        as.integer(self$embed_dim)))
            outputs <- self$dense(concat_attention)
            outputs
        },
        get_config = function() {
            list(
                name = self$name
            )
        }
    )
    
    layer_transformer_block <- Layer(
        classname = "TransformerBlock", 
        initialize = function(embed_dim, num_heads, ff_dim, rate=0.1, ...) {
            super()$`__init__`(...)
            self$att <- layer_MHA(embed_dim=embed_dim, num_heads=num_heads)
            self$ffn <- keras_model_sequential() %>%
                layer_dense(units = ff_dim, activation = "relu") %>%
                layer_dense(units = embed_dim)
            self$layernorm1 <- layer_layer_normalization(epsilon = 1e-6)
            self$layernorm2 <- layer_layer_normalization(epsilon = 1e-6)
            self$dropout1 <- layer_dropout(rate = rate)
            self$dropout2 <- layer_dropout(rate = rate)
        },
        call = function(inputs, ...) {
            attn_output <- self$att(inputs)
            attn_output <- self$dropout1(attn_output)
            out1 <- self$layernorm1(inputs + attn_output)
            ffn_output <- self$ffn(out1)
            ffn_output <- self$dropout2(ffn_output)
            self$layernorm2(out1 + ffn_output)
        },
        get_config = function() {
            list(
                name = self$name
            )
        }
    )
    
    if (!is.null(globalenv()$embed_dim)) old_embed_dim <- embed_dim
    if (!is.null(globalenv()$num_heads)) old_num_heads <- num_heads
    if (!is.null(globalenv()$ff_dim)) old_ff_dim <- ff_dim
    
    assign("embed_dim", embed_dim, envir = globalenv())
    assign("num_heads", num_heads, envir = globalenv())
    assign("ff_dim", ff_dim, envir = globalenv())
    
    for (i in seq_len(num_transformer_blocks)) {
        x <- layer_transformer_block(x,
                                    embed_dim = embed_dim,
                                    num_heads = num_heads,
                                    ff_dim = ff_dim)
    }
    if (!exists("old_embed_dim")) rm(embed_dim, envir = globalenv())
    if (!exists("old_num_heads")) rm(num_heads, envir = globalenv())
    if (!exists("old_ff_dim")) rm(ff_dim, envir = globalenv())
    x
}
