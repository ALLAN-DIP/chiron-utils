from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np


def classification(prompt, device):
    # Load the model and tokenizer, and place the model on the specified device
    classification_model = AutoModelForSequenceClassification.from_pretrained(
        "AutonLabTruth/llama3_m", torch_dtype=torch.float16
    ).to(device)
    classification_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

    # Tokenize the input and move it to the same device as the model
    inputs = classification_tokenizer(prompt, return_tensors="pt").to(device)

    # Set model to evaluation mode
    classification_model.eval()

    # Perform inference
    with torch.no_grad():
        logits = classification_model(**inputs).logits

    # Move logits back to CPU to avoid device mismatch issues
    logits = logits.cpu().numpy()

    # Calculate softmax and classification score
    softmaxed = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    score = softmaxed[0, 1]  # Assuming binary classification (index 1 is "untrustworthy")

    # Classify based on threshold
    if score > 0.15:
        predicted_class = "untrustworthy"
    else:
        predicted_class = "trustworthy"

    return predicted_class


# Define device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define prompt
prompt = (
    "FRANCE sends AUSTRIA: I think it's the most likely explanation for A Warsaw. If Russia was worried about Turkey, "
    "wouldn't they build A Moscow? Likewise, if Turkey intended to take Sevastopol. Wouldn't A Smyrna make more sense "
    "to capture Armenia and then Sev?"
)

# Call classification function and print result
predicted_class = classification(prompt, device)
print(predicted_class)
print(torch.cuda.memory_summary(device=device))

# reversed_closest_8_messages = [{'message':'1','sender':'AUS'},
#                                {'message':'2','sender':'ENG'},
#                                {'message':'3','sender':'ENG'},
#                                {'message':'4','sender':'AUS'},
#                                {'message':'5','sender':'AUS'},
#                                {'message':'6','sender':'ENG'}
#                                ]
# my_message = ""
# i = 0
# while i < len(reversed_closest_8_messages):
#     my_message += f"Message from {reversed_closest_8_messages[i]['sender']}:'{reversed_closest_8_messages[i]['message']}' "
#     if (
#         i == len(reversed_closest_8_messages) - 1
#         and reversed_closest_8_messages[i]['sender'] != 'AUS'
#     ):
#         my_message = my_message.rsplit(
#             f"Message from {reversed_closest_8_messages[i]['sender']}:'{reversed_closest_8_messages[i]['message']}' ",
#             1,
#         )[0]
#         reversed_closest_8_messages.pop()
#         i -= 1
#     i += 1

# print(my_message)
