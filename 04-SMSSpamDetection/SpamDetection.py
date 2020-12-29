#Script feito no windows, para executar só usar Python3
import pandas as pd
import re

train = pd.read_csv('TrainingSet/sms-hamspam-train.csv', sep='\t', header=None, names=['TAG', 'SMS'])

#Aqui retiro qualquer pontuação dos sms, converto tudo pra lower case e temparo o sms por palavras
train['SMS'] = train['SMS'].str.replace(r'[^\w\s]', '')
train['SMS'] = train['SMS'].str.lower()
train['SMS'] = train['SMS'].str.split()

#Crio uma lista com todas as palvras
words = []
for sms in train['SMS']:        
    for word in sms:
        words.append(word)

words = list(set(words))

#Crio uma lista onde cada plavra tem uma lista de 0 do tamanho da quantidade de sms
words_sms = {u_word: [0] * len(train['SMS']) for u_word in words}

#Incremento a lista de 0 para aparição daquela palavra para cada sms
for index, sms in enumerate(train['SMS']):
    for word in sms:
        words_sms[word][index] += 1

#Criação do dataset com os sms junto da lista de palavras
word_counts = pd.DataFrame(words_sms)
train_clean = pd.concat([train, word_counts], axis=1)

#Separando o dataset nos classificados como spam e ham
spams = train_clean[train_clean['TAG'] == 'spam']
hams = train_clean[train_clean['TAG'] == 'ham']

'''
    Começarei a criar as contantes do algoritimo que irei usar 'Naive Bayes',
    precisames do P e N para cada classe, N para as palavras e um alpha que iremos usar 1
'''
p_spam = len(spams) / len(train_clean)
p_ham = len(hams) / len(train_clean)

words_spam_message = spams['SMS'].apply(len)
n_spam = words_spam_message.sum()
words_ham_message = hams['SMS'].apply(len)
n_ham = words_ham_message.sum()

n_words = len(words)
alpha = 1

#Criamos a lista inicial de parametros com 0
parameters_spam = {u_word:0 for u_word in words}
parameters_ham = {u_word:0 for u_word in words}

#Usando o Naive Bayes calculamos os parametros para que possam serem usados depois no teste
for word in words:
    n_word_given_spam = spams[word].sum() # spam_messages already defined
    p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_words)
    parameters_spam[word] = p_word_given_spam

    n_word_given_ham = hams[word].sum() # ham_messages already defined
    p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_words)
    parameters_ham[word] = p_word_given_ham

'''
    Função para classificar dataset de teste
'''
def classify(message):
    message = re.sub(r'[^\w\s]', '', message)
    message = message.lower().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    
    for word in message:
        if word in parameters_spam:
           p_spam_given_message *= parameters_spam[word]
         
        if word in parameters_ham:
           p_ham_given_message *= parameters_ham[word]
    
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    else:
        return 'spam'

test = pd.read_csv('TestSet/sms-hamspam-test.csv', sep='\t', header=None, names=['SMS'])
test['TAG'] = test['SMS'].apply(classify)
test.to_csv('TestSet/sms-hamspam-test-resulte.csv')
