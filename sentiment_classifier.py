from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# Set parameters
vocab_size = 10000
maxlen = 200

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Build the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=16, input_length=maxlen),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=512, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Load word index
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Function to encode new reviews
def encode_review(text):
    tokens = text.lower().split()
    encoded = [1]  # <START> token
    for word in tokens:
        encoded.append(word_index.get(word, 2))  # 2 is <UNK> token
    return pad_sequences([encoded], maxlen=maxlen)

# Real-time prediction loop
while True:
    user_input = input("\nEnter your movie review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    
    encoded = encode_review(user_input)
    prediction = model.predict(encoded)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    print(f"Predicted Sentiment: {sentiment} (Confidence: {prediction:.4f})")
