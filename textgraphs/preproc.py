import re
import warnings
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

from baseline_tfidf import read_explanations

np.random.seed(42)
nltk.download("stopwords")


class DefaultLemmatizer:
  """
  Works best to transform texts before and also get lemmas during tokenization
  """
  def __init__(self, path_data: Path = None) -> None:
    if path_data is None:
      self.word2lemma = {}
    else:
      path_anno = path_data.joinpath("annotation")
      path = path_anno.joinpath("lemmatization-en.txt")

      def read_csv(_path: str, names: list = None) -> pd.DataFrame:
        return pd.read_csv(_path, header=None, sep="\t", names=names)

      df = read_csv(str(path), ["lemma", "word"])
      path_extra = path_anno.joinpath(
          "expl-tablestore-export-2017-08-25-230344/tables/LemmatizerAdditions.tsv"
      )
      df_extra = read_csv(str(path_extra), ["lemma", "word", "useless"])
      df_extra.drop(columns=["useless"], inplace=True)
      df_extra.dropna(inplace=True)

      length_old = len(df)
      # df = pd.concat([df, df_extra])  # Actually concat extra hurts MAP (0.462->0.456)
      print(
          f"Default lemmatizer ({length_old}) concatenated (or not) with extras ({len(df_extra)}) -> {len(df)}"
      )

      lemmas = df.lemma.tolist()
      words = df.word.tolist()

      def only_alpha(text: str) -> str:
        # Remove punct eg dry-clean -> dryclean so
        # they won't get split by downstream tokenizers
        return "".join([c for c in text if c.isalpha()])

      self.word2lemma = {
          words[i].lower(): only_alpha(lemmas[i]).lower()
          for i in range(len(words))
      }

  def transform(self, raw_texts: list) -> list:
    def _transform(text: str):
      return " ".join(
          [self.word2lemma.get(word) or word for word in text.split()])

    return [_transform(text) for text in raw_texts]


def preprocess_texts(
    texts: list,
    path_data: Path = None,
) -> (list, list):
  # NLTK tokenizer on par with spacy and less complicated
  tokenizer = nltk.tokenize.TreebankWordTokenizer()
  default_lemmatizer = DefaultLemmatizer(path_data)
  # wordnet_lemmatizer doesn't help
  texts = default_lemmatizer.transform(texts)
  stops = set(nltk.corpus.stopwords.words("english"))

  def lemmatize(token):
    return default_lemmatizer.word2lemma.get(token) or token

  def process(
      text: str,
      _tokenizer: nltk.tokenize.TreebankWordTokenizer,
  ) -> (list, list):
    _tokens = _tokenizer.tokenize(text.lower())
    _lemmas = [
        lemmatize(_tok) for _tok in _tokens
        if _tok not in stops and not _tok.isspace()
    ]
    return _tokens, _lemmas

  tokens, lemmas = zip(*[process(text, tokenizer) for text in tqdm(texts)])
  return tokens, lemmas


# # Spacy tokenizer
# def preprocess_texts(
#     texts: list,
#     remove_stop: bool = True,
#     remove_punct: bool = True,
#     only_alpha: bool = True,
#     path_data: Path = None,
# ) -> (list, list):
#   """
#   Get tokens and lemmas from texts
#   """
#   def prepreprocess(_texts):
#     """
#     Sometimes spacy doesn't handle punctuation well eg "work;life"
#     But completely removing all punctuation worsens score
#     """
#     return [text.replace(";", "; ").lower() for text in _texts]
#
#   # NLTK stopwords -> better MAP (see ranking/main)
#   stops = set(nltk.corpus.stopwords.words("english"))
#   # OTOH, NLTK WordNetLemmatizer is significantly less beneficial
#
#   texts = prepreprocess(texts)
#
#   default_lemmatizer = DefaultLemmatizer(path_data)
#   texts = default_lemmatizer.transform(texts)
#
#   nlp = spacy.load("en_core_web_sm")
#   tokens = []
#   lemmas = []
#   # Enabling spacy components is still fast but reduces MAP
#   for doc in tqdm(nlp.pipe(texts, disable=["ner", "tagger", "parser"])):
#     _tokens = []
#     _lemmas = []
#     for token in doc:
#       if token.text in stops and remove_stop:
#         # Better to disable tfidf stopwords and rely on this
#         continue
#       # if token.is_stop and remove_stop:
#       #     # Disabling slightly helps TF-IDF which has its own stop list
#       #     continue
#       if token.is_punct and remove_punct:
#         continue
#       if token.is_space:
#         continue  # Reduce useless nodes
#       if not token.is_alpha and only_alpha:
#         continue
#       # Lowercase to reduce noise
#       _tokens.append(token.text.lower())
#       # _lemmas.append(token.lemma_.lower())
#       _lemmas.append((default_lemmatizer.word2lemma.get(token.text)
#                       or token.lemma_).lower())
#     tokens.append(_tokens)
#     lemmas.append(_lemmas)
#
#   return tokens, lemmas


def test_preprocess_texts():
  print(
      preprocess_texts([
          "Which of these will most likely increase?",
          "Habitats support animals."
      ]))


def read_explanations_with_permutations(path):
  """
  Some explanations contain "combos" (eg man is male;human;organism)
  splitting them into (man is male, man is human, man is organism) is better
  The uids of the combos are suffixed with their combo number (eg originaluid_1)
  Improves score by ~1% but introduces significant code complexity
  """
  df = pd.read_csv(path, sep='\t')

  header, uid_column = [], None
  for name in df.columns:
    if name.startswith('[SKIP]'):
      if 'UID' in name and not uid_column:
        uid_column = name  # This is the column header
    else:
      header.append(name)  # These are all those not market '[SKIP]'

  if not uid_column or len(df) == 0:
    warnings.warn('Possibly misformatted file: ' + path)
    return []

  def format_uid(uid_orig, idx_combo):
    """
    Currently the rest of the code heavily relies on one-to-one mapping
    of uids, so duplicates are undesirable. For now add a unique suffix to
    each combo's uid that can be easily resolved at evaluation time
    """
    assert "_" not in uid_orig
    return f"{uid_orig}_{idx_combo}"

  arr = []
  for idx in df.index:
    # Original entry
    orig = ' '.join(
        str(s) for s in list(df.iloc[idx][header]) if not pd.isna(s))
    uid = df.at[idx, uid_column]
    arr.append([uid, orig, []])

    # Combo variants
    cells, combos, combo_tot = dict(), [], 1
    for h, v in zip(header, df.loc[idx][header]):
      s = '' if pd.isna(v) else str(v)  # cast to string in case of numbers
      options = [o.strip() for o in s.split(';')]
      cells[h] = options
      combos += [len(options)]
      combo_tot *= len(
          options)  # Count up the number of combos this contributes

    for i in range(combo_tot):
      # Go through all the columns, figuring out which combo we're on
      combo, lemmas, residual = [], [], i
      for j, h in enumerate(header):
        # Find the relevant part for this specific combo
        c = cells[h][residual % combos[j]]  # Works even if only 1 combo
        if len(cells[h]) > 1:
          lemmas.append(c)  # This is when there are choices
        combo.append(c)
        residual = residual // combos[j]  # TeeHee

      arr.append([
          format_uid(uid, i),  # uid
          ' '.join([c for c in combo if len(c) > 0]),  # text for this combo
          lemmas,
      ])
  return arr


def exp_skip_dep(
    path_exp: Path,
    col: str = "[SKIP] DEP",
    save_temp: bool = True,
) -> str:
  """
  Remove rows that have entries in deprecated column
  according to https://github.com/umanlp/tg2019task/issues/2
  """
  df = pd.read_csv(path_exp, sep="\t")
  if col in df.columns:
    df = df[df[col].isna()]
  path_new = "temp.tsv" if save_temp else Path(path_exp).name
  df.to_csv(path_new, sep="\t", index=False)
  return path_new


def save_unique_phrases(path_exp: Path, save_temp: bool = True) -> str:
  """
  The explanation tables have phrases in each column, normally we just
  concatenate everything into a sentence. However, maybe each unique
  phrases is important, so we concatenate eg "a kind of" -> "akindof"
  Edit: No this doesn't help, map -3% (46 -> 43)
  """
  df = pd.read_csv(path_exp, sep="\t")

  def transform(phrase: str) -> str:
    if type(phrase) != str:
      return phrase
    phrase_word = "".join([c for c in phrase if c.isalpha()])
    if phrase_word == phrase:
      return phrase
    return phrase + " " + phrase_word

  cols = [col for col in df.columns if "SKIP" not in col]  #
  df[cols] = df[cols].applymap(transform)
  path_new = "temp.tsv" if save_temp else Path(path_exp).name
  df.to_csv(path_new, sep="\t", index=False)
  return path_new


def get_df_explanations(
    path_tables: str,
    explanation_combos: bool,
    path_data: Path = None,
):
  """
  Make a dataframe of explanation sentences (~5000)
  """
  explanations = []
  columns = None
  for p in Path(path_tables).iterdir():
    columns = ["uid", "text"]
    p = exp_skip_dep(p)
    # p = save_unique_phrases(Path(p))
    if explanation_combos:
      explanations += read_explanations_with_permutations(str(p))
      columns.append("musthave")
    else:
      explanations += read_explanations(str(p))
  df = pd.DataFrame(explanations, columns=columns)
  df = df.drop_duplicates("uid").reset_index(drop=True)  # 3 duplicate uids
  tokens, lemmas = preprocess_texts(df.text.tolist(), path_data=path_data)
  df["tokens"], df["lemmas"], df["embedding"] = tokens, lemmas, None
  print("Explanation df shape:", df.shape)
  return df


def extract_explanation(exp_string):
  """
  Convert raw string (eg "uid1|role1 uid2|role2" -> [uid1, uid2], [role1, role2])
  """
  if type(exp_string) != str:
    return [], []
  uids = []
  roles = []
  for uid_and_role in exp_string.split():
    uid, role = uid_and_role.split("|")
    uids.append(uid)
    roles.append(role)
  return uids, roles


def split_question(q_string):
  """
  Split on option parentheses (eg "Question (A) option1 (B) option2" -> [Question, option 1, option2])
  Note that some questions have more or less than 4 options
  """
  return re.compile("\\(.\\)").split(q_string)


def test_split_question():
  print(
      split_question(
          'Which process? (A) flying (B) talking (C) seeing (D) reproducing (E) something'
      ))
  print(
      split_question(
          'Which process? (A) flying (B) talking (C) seeing (D) reproducing'))
  print(split_question('Which process? (A) flying (B) talking (C) seeing'))


def add_q_reformat(df: pd.DataFrame) -> pd.DataFrame:
  q_reformat = []
  questions = df.Question.values
  answers = df["AnswerKey.1"].values
  char2idx = {char: idx for idx, char in enumerate(list("ABCDE"))}

  for i in range(len(df)):
    q, *options = split_question(questions[i])
    idx_option = char2idx[answers[i]]
    q_reformat.append(" ".join([q.strip(), options[idx_option].strip()]))
  df["q_reformat"] = q_reformat
  return df


def get_questions(
    path: str,
    uid2idx: dict = None,
    path_data: Path = None,
) -> pd.DataFrame:
  """
  Identify correct answer text and filter out wrong distractors from question string
  Get tokens and lemmas
  Get explanation sentence ids and roles
  """
  # Dropping questions without explanations hurts score
  df = pd.read_csv(path, sep="\t")
  df = add_q_reformat(df)

  # Preprocess texts
  tokens, lemmas = preprocess_texts(df.q_reformat.tolist(), path_data)
  df["tokens"], df["lemmas"], df["embedding"] = tokens, lemmas, None

  # Get explanation uids and roles
  exp_uids = []
  exp_roles = []
  exp_idxs = []
  for exp_string in df.explanation.values:
    _uids, _roles = extract_explanation(exp_string)
    uids = []
    roles = []
    idxs = []
    assert len(_uids) == len(_roles)
    for i in range(len(_uids)):
      if _uids[i] not in uid2idx:
        continue
      uids.append(_uids[i])
      roles.append(_roles[i])
      idxs.append(uid2idx[_uids[i]])
    exp_uids.append(uids)
    exp_roles.append(roles)
    exp_idxs.append(idxs)
  df["exp_uids"], df["exp_roles"], df[
      "exp_idxs"] = exp_uids, exp_roles, exp_idxs

  print(df.shape)
  return df


# Build lemma dataframe where each unique lemma represents a node


def flatten(nested_list: list) -> list:
  return [item for lst in nested_list for item in lst]


def get_flattened_items(dfs: list, field: str) -> list:
  """
  Aggregate all items from a particular column from a list of dataframes
  """
  all_items = []
  for df in dfs:
    all_items.extend(flatten(df[field]))
  print(len(all_items))
  return all_items


def get_node_fns(dfs: list) -> (list, dict):
  all_lemmas = get_flattened_items(dfs, "lemmas")
  unique_lemmas = sorted(list(set(all_lemmas)))
  print(f"Total lemmas: {len(all_lemmas)}, unique: {len(unique_lemmas)}")

  lemma2node = {lemma: idx for idx, lemma in enumerate(unique_lemmas)}
  return unique_lemmas, lemma2node


def add_nodes(df, lemma2id):
  def get_nodes(lemmas):
    """
    Extract lemma node ids for every sentence
    """
    return [lemma2id[lemma] for lemma in lemmas]

  df["nodes"] = df.lemmas.apply(get_nodes)


def test_nodes(df, df_lemma, i=0):
  print(df.q_reformat.iloc[i])
  print(df_lemma.iloc[df.nodes.iloc[i]])


def embed_texts(texts):
  """
  Use universal encoder (transformer) from tf hub for sentence/word(?) embeddings
  Wrap with Keras model for convenient batching and progress bar
  Adapted from "Keras + Universal Sentence Encoder = Transfer Learning for text data"
  """
  texts = np.asarray(texts)
  tf.keras.backend.clear_session()

  module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
  embed = hub.Module(module_url)

  def get_universal_embedding(x):
    return embed(
        tf.squeeze(tf.cast(x, tf.string)),
        signature="default",
        as_dict=True,
    )["default"]

  inp = tf.keras.layers.Input(shape=(1, ), dtype=tf.string)
  out = tf.keras.layers.Lambda(get_universal_embedding)(inp)
  model = tf.keras.Model(inp, out)

  with tf.Session() as sess:
    tf.keras.backend.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    embeds = model.predict(texts, batch_size=128, verbose=True)

  tf.keras.backend.clear_session()  # avoid session closed error
  return embeds


def test_embed_texts():
  print(embed_texts(["hello", "bye"]))


def add_embeddings(dfs, fields_text):
  """
  Run embedding model predict over all texts combined to save time
  """
  assert len(dfs) == len(fields_text)
  lengths = [len(df) for df in dfs]
  nested_texts = [dfs[i][fields_text[i]].tolist() for i in range(len(dfs))]
  texts = [line for texts in nested_texts for line in texts]
  texts = [(" ".join(x) if type(x) == list else x) for x in texts]
  embeds = embed_texts(texts)
  idxs_split = np.cumsum(lengths)[:-1]  # 3 idxs -> split into 4
  embeds_split = np.split(embeds, idxs_split)
  assert len(embeds_split) == len(dfs)
  assert all([len(embeds_split[i]) == len(dfs[i]) for i in range(len(dfs))])
  for i in range(len(dfs)):
    # df must have an "embedding" column created first
    dfs[i].embedding = list(embeds_split[i])
  return dfs


def add_nn(df, n=100, annoy_index=None):
  """
  Add nearest neighbour explanation ids for each embedding
  """
  df["nn_exp"] = df.embedding.apply(
      lambda emb: annoy_index.get_nns_by_vector(emb, n))


def test_annoy(df, df_exp, i=0):
  print(df.q_reformat.iloc[i])
  print(df_exp.iloc[df.nn_exp.iloc[i]].text.values[:3])


def write_embs(df, field_text, field_emb="embedding"):
  """
  Make files to visualize explanation sentence embeddings in https://projector.tensorflow.org/
  """
  metas = df[field_text].apply(lambda x: "_".join(x) if type(x) == list else x)
  embs = df[field_emb].tolist()
  df_embs = pd.DataFrame(np.stack(embs))
  assert len(metas) == len(df_embs)

  with open("metas.tsv", "w") as file:
    file.write("\n".join(metas))
    file.write("\n")
  df_embs.to_csv("embs.tsv", sep="\t", header=False, index=False)
