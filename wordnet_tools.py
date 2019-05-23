import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import nltk
from nltk.corpus import wordnet as wn

import matplotlib.pyplot as plt
nltk.download('wordnet')
nltk.download('punkt')

def get_mammal_tree():
    mammal = wn.synset('mammal.n.01')
    print(mammal.definition())
    print(mammal.hyponyms())
    mammal_names = []
    mammal_descriptions = []
    mammal_descriptions_tokens = []
    tree_structure = {}
    print(wn.synsets('animal'))

    # depth 6 of mammal subtree
    tree_structure[mammal.name()] = len(mammal.hyponyms())
    for h in mammal.hyponyms():
        print(h.name(), h.definition())
        mammal_names.append(h.name())
        mammal_descriptions.append(h.definition())
        mammal_descriptions_tokens.append(nltk.word_tokenize(h.definition()))
        tree_structure[h.name()] = len(h.hyponyms())
        for hn in h.hyponyms():
            print('>',hn.name(),hn.definition())
            mammal_names.append(hn.name())
            mammal_descriptions.append(hn.definition())
            mammal_descriptions_tokens.append(nltk.word_tokenize(hn.definition()))
            tree_structure[hn.name()] = len(hn.hyponyms())
            for hnn in hn.hyponyms():
                print('>>',hnn.name(),hnn.definition())
                mammal_names.append(hnn.name())
                mammal_descriptions.append(hnn.definition())
                mammal_descriptions_tokens.append(nltk.word_tokenize(hnn.definition()))
                tree_structure[hnn.name()] = len(hnn.hyponyms())
                for hnnn in hnn.hyponyms():
                    print('>>>',hnnn.name(),hnnn.definition())
                    mammal_names.append(hnnn.name())
                    mammal_descriptions.append(hnnn.definition())
                    mammal_descriptions_tokens.append(nltk.word_tokenize(hnnn.definition()))
                    tree_structure[hnnn.name()] = len(hnnn.hyponyms())
                    for hnnnn in hnnn.hyponyms():
                        print('>>>>',hnnnn.name(),hnnnn.definition())
                        mammal_names.append(hnnnn.name())
                        mammal_descriptions.append(hnnnn.definition())
                        mammal_descriptions_tokens.append(nltk.word_tokenize(hnnnn.definition()))
                        tree_structure[hnnnn.name()] = len(hnnnn.hyponyms())
                        for hnnnnn in hnnnn.hyponyms():
                            print('>>>>>',hnnnnn.name(),hnnnnn.definition())
                            mammal_names.append(hnnnnn.name())
                            mammal_descriptions.append(hnnnnn.definition())
                            mammal_descriptions_tokens.append(nltk.word_tokenize(hnnnnn.definition()))
                            
    return mammal_descriptions, mammal_descriptions_tokens, mammal_names, tree_structure

def gen_wordnet_embeddings(mammal_descriptions):
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
    nnlm = hub.Module("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1")
    nnlm_embeddings = nnlm(mammal_descriptions)
    print(nnlm_embeddings)
    sess = tf.Session()
    sess.run(tf.tables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    nnlm_np = sess.run(nnlm_embeddings)
    elmo_dict = elmo(mammal_descriptions,signature="default",as_dict=True)
    elmo_embeddings = elmo_dict['default']
    #print(embeddings.shape)
    elmo_np = sess.run(elmo_embeddings)

    np.save('./data/nnlm2.npy',nnlm_np)
    np.save('./data/elmo2.npy',elmo_np)
    
def load_wordnet_data(name,shuffle=True):
    embeddings = np.load('./data/'+name+'.npy')
    if shuffle:
        np.random.shuffle(embeddings)
    N = len(embeddings)
    split = np.int(np.floor(0.8*N))
    embeddings_train = embeddings[:split]
    embeddings_test = embeddings[split:]
    return embeddings_train, embeddings_test

def draw_line(parent,child,h,lw,clr):
    plt.scatter(parent[0], parent[1], c='b',marker='+',s=0.5)
    plt.text(parent[0]+0.01, parent[1]+0.01, h.name(), fontsize=9)
    plt.scatter(child[0], child[1], c='b',marker='+',s=0.5)
    plt.text(child[0]+0.01, child[1]+0.01, h.name(), fontsize=9)
    x = [parent[0],child[0]]
    y = [parent[1],child[1]]
    plt.plot(x,y, c=clr, linewidth=lw)

def draw_wordnet_tree(embeddings,limit):
    mammal = wn.synset('mammal.n.01')
    # connect i->j
    i = 0
    j = 1
    plt.figure(figsize=(24,24)) 
    color = ['r','g','y','m','b','c','k']

    parent = embeddings[i]
    plt.scatter(parent[0], parent[1], c='r',marker='+',s=10)
    plt.text(parent[0]+0.01, parent[1]+0.01, mammal.name(), fontsize=9)
    for h in mammal.hyponyms():
        child = embeddings[j]
        draw_line(parent,child,h,3,color[0])
        parent2 = child
        j+=1
        for hn in h.hyponyms():
            child2 = embeddings[j]
            draw_line(parent2,child2,hn,2.5,color[1])
            parent3 = child2
            j+=1
            for hnn in hn.hyponyms():
                child3 = embeddings[j]
                draw_line(parent3,child3,hnn,2,color[2])
                parent4 = child3
                j+=1
                for hnnn in hnn.hyponyms():
                    child4 = embeddings[j]
                    draw_line(parent4,child4,hnnn,1.5,color[3])
                    parent5 = child4
                    j+=1
                    for hnnnn in hnnn.hyponyms():
                        child5 = embeddings[j]
                        draw_line(parent5,child5,hnnnn,1,color[4])
                        parent6 = child5
                        j+=1
                        for hnnnnn in hnnnn.hyponyms():
                            child6 = embeddings[j]
                            draw_line(parent2,child2,hnnnnn,0.5,color[5])
                            j+=1
                            if j>limit:
                                ax = plt.gca()
                                circle = plt.Circle((0, 0), 1.0, color='k', fill=False)
                                ax.add_artist(circle)
                                plt.axis('equal')
                                plt.grid()
                                plt.show()
                                return 0



                            