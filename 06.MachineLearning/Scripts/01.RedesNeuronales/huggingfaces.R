library("text")
textrpp_install()
textrpp_initialize(save_profile = TRUE)

textGeneration("I live in Madrid and I work",
               model = "gpt2")



install.packages("sentencepiece")
library(sentencepiece)
textClassify(
  x = "Usar modelos preentrenados es muy fácil",
  model = ""
)


textClassify(
  x = c("Todos los edificios de mi barrio son de colores"))


textZeroShot(
  sequences = c("Almeida es el mejor","Espero que Pedro Sánchez repita como presidente","Irene Montero es una enchufada","Isabel Ayuso es una mentirosa"),
  candidate_labels = c("PP political party","PSOE political party"),
  hypothesis_template = "This example supports {}.",
  multi_label = F,
  model = "facebook/bart-large-mnli",
  device = "cpu",
  tokenizer_parallelism = FALSE,
  logging_level = "error",
  return_incorrect_results = FALSE,
  set_seed = 202208L
)


library(rtweet)


texto <- search_tweets2("Goya AND Alterio", n = 20)

texto <- as.data.frame(texto$full_text)

cod <- textZeroShot(
  sequences = as.vector(texto$`texto$full_text`),
  candidate_labels = c("Goya","Alterio"),
  hypothesis_template = "This example talks about {}.",
  multi_label = F,
  model = "facebook/bart-large-mnli",
  device = "cpu",
  tokenizer_parallelism = FALSE,
  logging_level = "error",
  return_incorrect_results = FALSE,
  set_seed = 202208L
)

table(cod$labels_x_1)
quantile(cod$scores_x_1)
cod_2 <- cod[cod$scores_x_1>0.71,]


########################## EMBEDDINGS #########################

texts <- c("I'm feeling relatedness with others","That's great!")

#construir embeddings para darle valores a las palabras y posiciones en escenarios multidimensionales.

embeddings <-  textEmbed(texts)

embeddings$tokens
embeddings$texts

# conocer qué modelos hay descargardos y borrarlos

textModels()

textModelsRemove()

# Look at example data included in the text-package comprising both text and numerical
# variables (note that there are only 40 participants in this example).

Language_based_assessment_data_8

# Transform the text/word data to word embeddings (see help(textEmbed) to see the default
#settings).

word_embeddings <- textEmbed(Language_based_assessment_data_8,
                             model = "bert-base-uncased",
                             aggregation_from_layers_to_tokens = "concatenate",
                             aggregation_from_tokens_to_texts = "mean",
                             keep_token_embeddings = FALSE)
# See how the word embeddings are structured
word_embeddings
# Save the word embeddings to avoid having to embed the text again. It is good practice
#to save output from analyses that take a lot of time to compute, which is often the case
#when analyzing text data.

setwd()

saveRDS(word_embeddings, "word_embeddings.rds")

# Get the saved word embeddings (again)

word_embeddings <- readRDS("word_embeddings.rds")