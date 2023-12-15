import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Function to calculate the Title feature (F1) for a given sentence.
def title_feature(sentence, title):
    # Tokenize the title and the sentence
    title_words = title.split()
    sentence_words = sentence.split()

    # Count how many title words appear in the sentence
    matches = sum(1 for word in title_words if word in sentence_words)

    # Calculate the feature score
    feature_score = matches / len(title_words)

    # Return the score
    return min(max(feature_score, 0), 1)

# Function to calculate the Sentence Length feature (F2) for a given sentence.

def sentence_length_feature(sentence, longest_sentence):
    # Tokenize the sentences to get the number of words
    num_words_sentence = len(sentence.split())
    num_words_longest_sentence = len(longest_sentence.split())

    # Calculate the feature score
    feature_score = num_words_sentence / num_words_longest_sentence

    return min(max(feature_score, 0), 1)

# Function to calculate the Sentence Position feature (F3) for a given sentence position.

def sentence_position_feature(position, total_sentences):
    # Calculate the two relations
    relation1 = 1 / (position + 1)
    relation2 = 1 / (total_sentences - position)

    # Calculate the feature score
    feature_score = max(relation1, relation2)

    return min(max(feature_score, 0), 1)


# Function to calculate the Term Weight feature (F5) for a given sentence.

def term_weight_feature(sentence, entire_text):
    # Tokenize the sentence and entire text to get words
    sentence_words = sentence.split()
    entire_text_words = entire_text.split()

    # Compute the frequencies in the sentence
    frequency_in_sentence = sum(sentence_words.count(word) for word in set(sentence_words))

    # Compute the frequencies in the entire text
    frequency_in_text = sum(entire_text_words.count(word) for word in set(sentence_words))

    # Calculate the feature score
    feature_score = frequency_in_sentence / frequency_in_text if frequency_in_text != 0 else 0

    # Ensure the score is in the interval [0, 1]
    return min(max(feature_score, 0), 1)


# Function to calculate the Proper Noun feature (F6) for a given sentence.

def proper_noun_feature(sentence):
    # Tokenize the sentence
    words = word_tokenize(sentence)

    # Get the part-of-speech tags
    tagged_words = pos_tag(words)

    # Count the number of proper nouns
    num_proper_nouns = sum(1 for word, tag in tagged_words if tag == 'NNP' or tag == 'NNPS')

    # Calculate the feature score
    feature_score = num_proper_nouns / len(words) if len(words) != 0 else 0

    return min(max(feature_score, 0), 1)



# Helper function to check if a word represents a float
def is_float(token):
    try:
        float(token)
        return True
    except ValueError:
        return False

# Function to calculate the Numerical Data feature (F7) for a given sentence.
def numerical_data_feature(sentence):

    # Tokenize the sentence
    words = word_tokenize(sentence)

    # Count the number of numerical data tokens
    num_numerical_data = sum(1 for word in words if word.isdigit() or is_float(word))

    # Calculate the feature score
    feature_score = num_numerical_data / len(words) if len(words) != 0 else 0

    return min(max(feature_score, 0), 1)


# Function to compute the combined score for a sentence based on multiple features.
def compute_sentence_score(sentence, entire_text, title, weights, longest_sentence, position, total_sentences):

    # Compute individual feature scores
    F1 = title_feature(sentence, title)
    F2 = sentence_length_feature(sentence, longest_sentence)
    F3 = sentence_position_feature(position, total_sentences)
    F5 = term_weight_feature(sentence, entire_text)
    F6 = proper_noun_feature(sentence)
    F7 = numerical_data_feature(sentence)

    # Combine the feature scores using the weights
    score = (weights['F1'] * F1 +
             weights['F2'] * F2 +
             weights['F3'] * F3 +
             weights['F5'] * F5 +
             weights['F6'] * F6 +
             weights['F7'] * F7)

    return score

def calculateLongestSent(sentences):
  count=0
  for sent in sentences:
    if len(sent) > count:
      count = len(sent)
      ls = sent
  return ls