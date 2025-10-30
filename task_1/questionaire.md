# Questionnaire for Task 1: Understanding Transformers and LLMs

## Important to note

Answer concisely and thoughtfully. Do not use LLMs to generate responses, it is acceptable if answers are imperfect.

### Question 1

This question is about the Feed-forward neural network (FFNN) block present in the transformer architecture. You've noticed that the FFNN has two types of layers, namely Linear and ReLU (besides LayerNorm and other layers, which we omit here). The neural network applies weights and biases in the linear layer, and ReLU layer introduces a non-linearity. Why do you think this non-linearity is needed? Are we better off not using ReLU (and thus saving some compute)? If not, are there any alternatives to ReLU that could be thought of? (they need not be more efficient, just alternatives are enough)

**Your answer**:
### Question 2

During your reading for the assignment, you come across quantization, a way to reduce the LLM model's size by expressing its weights as 16-bit FP numbers or 8-bit integers. You decide to apply quantization to the LLM you're working with, so that you can fit a larger LLM into a smaller GPU VRAM and get better responses. What could be the possible issues you face with this approach? 

**Your answer**:
### Question 3

You've read about the attention mechanism for this assignment. Current LLM models implement a lot of attention heads (multi-headed attention or MHA) instead of having one big attention head. Why do you think this is the case?

**Your answer**:

### Question 4 & 5 Case Study

Q4 - Q5 are based on the following case study.

Pranav and Soham are new to machine learning, and have recently learned about transformers and neural networks. They want to start off by implementing a neural network that can help identify handwritten digits, with training data obtained from the MNIST dataset. They decide to have a neural network architecture that looks like this:

28 x 28 image -> 784-dimensional vector -> passes through a layer of 784 neurons and compresses it to 256 dimensions -> apply ReLU -> Pass through another layer, compressing it to 128 dimensions -> ReLU -> Another layer that compresses it to 10 dimensions -> apply softmax -> Make a prediction

### Question 4

They've now decided to begin training on the dataset, and split it into train, validation and test sets. For the loss function, they'll be using softmax regression, and optimizer would be the SGD optimizer. Pranav decides that the learning rate should be set to 1 so that the training ends quicker, and begins training the NN. However, the training seems to continue for a long time, with no significant improvements in both training and validation accuracy. Moreover, the values for accuracy seem to be oscillatory (eg: the accuracy would jump from 56% to 59%, and back to 56% again after successive iterations). Why do you think this could be the case?

**Your answer**:
### Question 5

Thanks to your help, they were able to figure out the problem! However, their NN seems to perform very poorly (~60% accuracy) on the prediction task. Soham suggests that adding more neurons to the intermediate layers would help the model capture relationships better. Using this suggestion, Pranav and Soham notice that the model's performance on the training set has improved, but the model is performing poorly on the validation set. Why has this happened, and how could this be mitigated?

**Your answer**:
### Question 6 & 7 Case Study

Q6-Q7 are based on the following case study.

Vimarsh is hosting a LLM called DaSHGPT on his small cluster of GPUs for about 70 people. The LLM uses the classic transformer architecture that you've learnt about (emdedding, attention heads, FFNN, softmax). Vimarsh also knows that GPUs are good at processing things in parallel, and decides to leverage this by parallelizing the pipeline as much as he can.

### Question 6

What are some parallelization techniques he could use to speed up inference on his small GPU cluster?

**Your answer**:
### Question 7

Vimarsh recently came across what is known as "batching" prompts together. He decides to save compute and memory resources, and sets the batch size to 64. This approach works well, as a lot of compute power is saved due to GPUs working less frequently, and in parallel. However, users complain of long waiting time after giving a prompt to DaSHGPT. Why has this happened? What would you do to fix this?

**Your answer**:
### Question 8, 9 & 10 Case Study

Q8 - Q10 are based on the following case study.

Atharva has recently read the paper on attention. Thinking that "Attention is all He Needs", he begins working on his own transformer/LLM. He's able to code up almost all aspects of the neural networks, but needs help in some other ones. Could you help him?

### Question 8

Assume that you have a sentence that is of the form "The quick brown fox jumps over the lazy ___". You want the LLM to predict the next token in the sequence. Atharva's LLM does this by tokenizing the input sentence (assume one token = one word in the sentence), pushing it into the transformer's attention layers, computes all the attention dot products for the sentence, then multiplies it by the value matrix, finds the linear combination and updates the embedded vector, and finally passes it through the FFNN, repeating this procedure from the beginning each time a new token is added, thus getting the result using softmax regression. However, Atharva's LLM seems to be very slow at producing the output. What could be the issue?

**Your answer**:
### Question 9

Atharva has realized his mistake now, and is set to correct it. He now stores the required computations in his GPU VRAM. Assuming that Atharva is using only 1 GPU with 16GB VRAM, a 7B parameter model with FP16 precision consumes about 14 GB of his VRAM. Happy that his LLM fits in the GPU, Atharva begins inference, but soon realises that the time taken to produce an output keeps increasing as he gives more and more prompts, eventually piling up to an unacceptable delay. Why has this happened? What is the fix that comes to your mind? Assume that the prompts are interdependent i.e. prompt 1 uses context from prompt 2, and so on. (Hint: Think in terms of where and how the pre-computed values are being stored and accessed. You ideally do not want the values to consume GPU VRAM).

**Your answer**:
### Question 10

Great, Atharva has listened to your advice and implemented "the fix" you've mentioned above. However, instead of an eventual, unacceptable delay, Atharva now has a fixed but noticeable delay in producing an output for each token of every prompt. Why has this happened? (Hint: think about where the KV cache is stored now, and what happens each time a new token is generated. Can this data movement be reduced?) Can you think of a simple way to reduce this delay?

**Your answer**:


### Question 11 (BONUS) 

To give you a small taste on what the LLM inference scheduling project is working on, here's a toy problem for you: (Do note that this is a very advanced problem for this assignment).

You have a 4-node setup:
C1, C2, C3 -> 1 GPU each, 8 GB VRAM.
C4 -> 2 GPUs (A1, A2), 12 GB VRAM each.
Network between nodes is slow (considerable communication delay).
NVLink is present within C4 (transfer b/w A1 -> A2) is fast.
You have a 3B parameter model in FP16 (6 GB VRAM per model instance).


Your task:
Design a scheduling algorithm to assign prompts to GPUs based on their total length (input + output), classified as short / medium / large. (Assume output length is known beforehand - a very unrealistic one, but fits for this case)
You may draw a graph or diagram to explain your approach.
Proper justification MUST be provided for your approach.


Hints:

Consider VRAM constraints: a GPU must fit the model + KV cache for the assigned prompt.
Prefer fast local connections for prompts that require multiple GPUs.
Assign short prompts to smaller GPUs, and long prompts to larger GPUs.
Aim for load balancing while minimizing communication overhead.

If you're able to figure out an optimal scheduling algorithm for the above case, list two problems you'd face if you were to drop the unrealistic assumption made above.


**Your answer**:
