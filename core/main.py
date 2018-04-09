import sys
import os
from subprocess import call
import numpy as np
import math
import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from datetime import datetime
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

from nltk.tokenize import sent_tokenize, word_tokenize

from os import listdir
from os.path import isfile, join
EREDir = "../nw/ere"
SRCDir = "../nw/source"
threshold = 0.1

erefiles = [f for f in listdir(EREDir) if isfile(join(EREDir, f))]
srcfiles = [f for f in listdir(SRCDir) if isfile(join(SRCDir, f))]

Sentences = {}

import xml.etree.ElementTree as ET
	
eventlist = []
for fname in erefiles:
	#print fname
	with open(join(EREDir, fname)) as f:
		content = "\n".join(f.readlines())
		content = content.replace('&', '&amp;')
		content = content.replace('.', '')
		root = ET.fromstring(content)
		
		for child in root:
			if child.tag=="hoppers":
				hopperschild = child
				for hopper in hopperschild:
					hopperid = hopper.attrib["id"]
					for event_mention in hopper:
						# A hopper can contain multiple event mentions - event coreference!
						event_type = event_mention.attrib["type"]
						event_subtype = event_mention.attrib["subtype"]
						action = ""
						agent_sub = ""
						agent_obj = ""
						location = ""
						time = ""
						for child in event_mention:
						
							if child.tag == "trigger":
								action = child.text
							elif child.attrib["role"] in ["agent", "instrument", "entity"]:
								agent_sub = child.text
							elif child.attrib["role"] in ["victim"]:
								agent_obj = child.text
							elif child.attrib["role"] in ["place", "destination"]:
								location = child.text
							elif child.attrib["role"] in ["time"]:
								time = child.text
						
						eventlist.append((hopperid, event_type+"-"+event_subtype, action, agent_sub, agent_obj, location, time))

#print eventlist						

#vectordata = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True) # path to Google W2V
vectordata = KeyedVectors.load_word2vec_format('../lexvec.enwiki+newscrawl.300d.W.pos.vectors', binary=False) # path to LexVec

#glove_file = '../glove.6B/glove.6B.300d.txt' # path to GloVe
#tmp_file = get_tmpfile("test_word2vec.txt")

#glove2word2vec(glove_file, tmp_file)
#vectordata = KeyedVectors.load_word2vec_format(tmp_file)


def vector_add(v1, v2):
	if len(v1)==len(v2):
		return [v1[i]+v2[i] for i in range(len(v1))]
	else:
		return 0
		

def vectorize(event_params):
	argvectors = []
	for arg in event_params:
		#print arg
		vector_sum = [0]*300
		l = 0
		for word in word_tokenize(arg):
			try:
				vector_sum = vector_add(vectordata.get_vector(word.lower()), vector_sum)
			except:
				print "Using a random vector for OOV word:", word.lower()
				vector_sum = vector_add(np.random.rand(300), vector_sum)
			l += 1
		
		if l==0:
			vector_avg = vector_sum
		else:
			vector_avg = [float(x)/l for x in vector_sum]
		argvectors.extend(vector_avg)
		break
	#print argvectors	
	return tuple(argvectors)
	
eventdict = {}
for event in eventlist:
	if eventdict.has_key(event[1]):
		eventdict[event[1]].append(vectorize(event[2:]))
	else:
		eventdict[event[1]] = [vectorize(event[2:])]
	
for etype in eventdict.keys():
	vector_sum = [0]*300
	l = 0
	for vec in eventdict[etype]:
		vector_sum = vector_add(vec, vector_sum)
		l += 1
	if l==0:
		vector_avg = vector_sum
	else:
		vector_avg = [float(x)/l for x in vector_sum]
	eventdict[etype] = vector_avg		

obsraweventlist = []
for fname in srcfiles:
	#print fname
	docstart = datetime.now()
	with open(join(SRCDir, fname)) as f:
		content = "".join(f.readlines())
		content = content.replace('&', '&amp;')
		root = ET.fromstring(content)
		text = ""
		for child in root:
			if child.tag=="TEXT" or child.tag=="HEADLINE":
				text += child.text
		with open("text.txt", "w") as out:
			out.write(text.encode('utf8'))
			out.close()
		os.chdir("../senna")
		os.system("./senna -srl < ../core/text.txt > ../core/tags.txt")
		os.chdir("../core")
	
	with open("tags.txt") as tags:
		lines = tags.readlines()
		tl = [[w.strip() for w in l.split("\t")] for l in lines]
		n = len(tl)
		blocks = [[]]
		count = 0
		for i in range(n):
			#print tl[i]
			if len(tl[i])>1:
				blocks[count].append(tl[i])
			elif i<n-1:
				blocks.append([])
				count += 1
		#print blocks[5]
		
	
	for i in range(len(blocks)):
		nrows = len(blocks[i])
		ncols = len(blocks[i][0])
		for col in range(2,ncols):
			action = ""
			agent_sub = ""
			agent_obj = ""
			location = ""
			time = ""
			row = 0
			while row<nrows:
				if blocks[i][row][col]=="S-V":
					action = blocks[i][row][0]
				if blocks[i][row][col]=="B-V":
					while blocks[i][row][col]!="E-V":
						action += " "+blocks[i][row][0]
						row += 1
					action += " "+blocks[i][row][0]
				if blocks[i][row][col]=="S-A0":
					agent_sub = blocks[i][row][0]
				if blocks[i][row][col]=="B-A0":
					while blocks[i][row][col]!="E-A0":
						agent_sub += " "+blocks[i][row][0]
						row += 1
					agent_sub += " "+blocks[i][row][0]
				if blocks[i][row][col]=="S-A1":
					agent_obj = blocks[i][row][0]
				if blocks[i][row][col]=="B-A1":
					while blocks[i][row][col]!="E-A1":
						agent_obj += " "+blocks[i][row][0]
						row += 1
					agent_obj += " "+blocks[i][row][0]
				if blocks[i][row][col]=="S-AM-LOC":
					location = blocks[i][row][0]
				if blocks[i][row][col]=="B-AM-LOC":
					while blocks[i][row][col]!="E-AM-LOC":
						location += " "+blocks[i][row][0]
						row += 1
					location += " "+blocks[i][row][0]
				if blocks[i][row][col]=="S-AM-TMP":
					time = blocks[i][row][0]
				if blocks[i][row][col]=="B-AM-TMP":
					while blocks[i][row][col]!="E-AM-TMP":
						time += " "+blocks[i][row][0]
						row += 1
					time += " "+blocks[i][row][0]
				row += 1
			obsraweventlist.append((action, agent_sub, agent_obj, location, time))	
	#break 

obseventlist = []	
for rawevent in obsraweventlist:
	vec = vectorize(rawevent)
	max_sim = 0
	for etype in eventdict.keys():
		temp = 1-spatial.distance.cosine(eventdict[etype], vec)
		if temp > max_sim:
			max_sim = temp
			cand_etype = etype
	obseventlist.append((cand_etype, rawevent[0], rawevent[1], rawevent[2], rawevent[3], rawevent[4]))

vectorized_eventlist = [(event[0], event[1], vectorize(event[2:])) for event in eventlist]
vectorized_obseventlist = [(event[0], vectorize(event[1:])) for event in obseventlist]

count_matchP = 0
count_matchR = 0
match_sim_threshold = 0.5
m = len(vectorized_obseventlist)
n = len(vectorized_eventlist)

for obsevent in vectorized_obseventlist:
	for event in vectorized_eventlist:
		if 1-spatial.distance.cosine(obsevent[1], event[2]) > match_sim_threshold:
			count_matchP += 1
			break
			
for event in vectorized_eventlist:
	for obsevent in vectorized_obseventlist:
		if 1-spatial.distance.cosine(obsevent[1], event[2]) > match_sim_threshold:
			count_matchR += 1
			break
			
precision = float(count_matchP)/m
recall = float(count_matchR)/n
print "Precision for event detection (span):", precision
print "Recall for event detection (span):", recall
print "F1 score for event detection (span):", 2*precision*recall/(precision+recall)

count_matchP = 0
count_matchR = 0
for obsevent in vectorized_obseventlist:
	for event in vectorized_eventlist:
		if event[1]==obsevent[0] and 1-spatial.distance.cosine(obsevent[1], event[2]) > match_sim_threshold:
			count_matchP += 1
			break
		
for event in vectorized_eventlist:
	for obsevent in vectorized_obseventlist:
		if event[1]==obsevent[0] and 1-spatial.distance.cosine(obsevent[1], event[2]) > match_sim_threshold:
			count_matchR += 1
			break
			
precision = float(count_matchP)/m
recall = float(count_matchR)/n
print "Precision for event detection (span+type):", precision
print "Recall for event detection (span+type):", recall
print "F1 score for event detection (span+type):", 2*precision*recall/(precision+recall)

pred_vectorized_eventlist = [[0, event[1], event[2]] for event in vectorized_eventlist]

match_coref_threshold = 0.9
cluster = 0
for i in range(n):
	clusterfound = False
	for j in range(i):
		if 1-spatial.distance.cosine(pred_vectorized_eventlist[j][2], pred_vectorized_eventlist[i][2])>match_coref_threshold:
			pred_vectorized_eventlist[i][0] = pred_vectorized_eventlist[j][0]
			clusterfound = True
			break
	if not clusterfound:
		cluster += 1
		pred_vectorized_eventlist[i][0] = cluster
	
gold_clusters = []
for i in range(n):
	clusterfound = False
	for j in range(len(gold_clusters)):
		if vectorized_eventlist[i][0] == vectorized_eventlist[gold_clusters[j][0]][0]:
			gold_clusters[j].append(i)
			clusterfound = True
			break
		
	if not clusterfound:
		gold_clusters.append([i])
		
pred_clusters = []
for i in range(n):
	clusterfound = False
	for j in range(len(pred_clusters)):
		if pred_vectorized_eventlist[i][0] == pred_vectorized_eventlist[pred_clusters[j][0]][0]:
			pred_clusters[j].append(i)
			clusterfound = True
			break
		
	if not clusterfound:
		pred_clusters.append([i])
	
MUCR_num = 0
MUCR_denom = 0
B3R_sum = 0		
for S in gold_clusters:
	partition = []
	for R in pred_clusters:
		l = [x for x in S if x in R]
		if len(l)>0:
			partition.append(l)
	pS = len(partition)
	MUCR_num += len(S) - pS
	MUCR_denom += len(S) - 1
	temp = 0
	for Pj in partition:
		temp += len(Pj)*(len(S)-len(Pj))
	B3R_sum += 1 - float(temp) / len(S)**2 
	
MUCR = float(MUCR_num)/MUCR_denom
B3R = B3R_sum/len(gold_clusters)

MUCP_num = 0
MUCP_denom = 0
B3P_sum = 0	
for R in pred_clusters:
	partition = []
	for S in gold_clusters:
		l = [x for x in R if x in S]
		if len(l)>0:
			partition.append(l)
	pR = len(partition)
	MUCP_num += len(R) - pR
	MUCP_denom += len(R) - 1
	temp = 0
	for Pj in partition:
		temp += len(Pj)*(len(R)-len(Pj))
	B3P_sum += 1 - float(temp) / len(R)**2
	
MUCP = float(MUCP_num)/MUCP_denom
B3P = B3P_sum/len(pred_clusters)

MUCF1 = 2*MUCP*MUCR/(MUCP+MUCR)
B3F1 = 2*B3P*B3R/(B3P+B3R)

print "MUC-Precision for event coreference:", MUCP
print "MUC-Recall for event coreference:", MUCR
print "MUC-F1 for event coreference:", MUCF1
print "B3-Precision for event coreference:", B3P
print "B3-Recall for event coreference:", B3R
print "B3-F1 for event coreference:", B3F1
	 
rc = 0
rn = 0
wc = 0
wn = 0
for i in range(n):
	for j in range(i+1,n):
		if pred_vectorized_eventlist[i][0] == pred_vectorized_eventlist[j][0] and vectorized_eventlist[i][0] == vectorized_eventlist[j][0]:
			rc += 1
		if pred_vectorized_eventlist[i][0] == pred_vectorized_eventlist[j][0] and vectorized_eventlist[i][0] != vectorized_eventlist[j][0]:
			wc += 1
		if pred_vectorized_eventlist[i][0] != pred_vectorized_eventlist[j][0] and vectorized_eventlist[i][0] != vectorized_eventlist[j][0]:
			rn += 1
		if pred_vectorized_eventlist[i][0] != pred_vectorized_eventlist[j][0] and vectorized_eventlist[i][0] == vectorized_eventlist[j][0]:
			wn += 1

Pc = float(rc)/(rc+wc)
Pn = float(rn)/(rn+wn)
Rc = float(rc)/(rc+wn)
Rn = float(rn)/(rn+wc)
Fc = 2*Pc*Rc/(Pc+Rc)
Fn = 2*Pn*Rn/(Pn+Rn)

BLANCP = (Pc+Pn)/2
BLANCR = (Rc+Rn)/2
BLANC = (Fc+Fn)/2

print "BLANC-Precision for event coreference:", BLANCP
print "BLANC-Recall for event coreference:", BLANCR
print "BLANC-F1 for event coreference:", BLANC

modS = len(gold_clusters)
modR = len(pred_clusters)

def phi(cluster1, cluster2):
	common = len([x for x in cluster1 if x in cluster2])
	return float(2*common)/(len(cluster1)+len(cluster2))

phi_g = 0	
if modS < modR:
	temp_pred = list(pred_clusters)
	for S in gold_clusters:
		max_sim = 0
		chosenindex = 0
		for R in temp_pred:
			if phi(S, R) > max_sim:
				max_sim = phi(S, R)
				chosenindex = temp_pred.index(R)
		del temp_pred[chosenindex]
		phi_g += max_sim
else:
	temp_gold = list(gold_clusters)
	for R in pred_clusters:
		max_sim = 0
		chosenindex = 0
		for S in temp_gold:
			if phi(S, R) > max_sim:
				max_sim = phi(S, R)
				chosenindex = temp_gold.index(S)
		del temp_gold[chosenindex]
		phi_g += max_sim

EntityCEAFP = float(phi_g) / modR
EntityCEAFR = float(phi_g) / modS
EntityCEAF = 2*EntityCEAFP*EntityCEAFR/(EntityCEAFP+EntityCEAFR)
print "Entity-based CEAF-Precision for event coreference:", EntityCEAFP
print "Entity-based CEAF-Recall for event coreference:", EntityCEAFR
print "Entity-based CEAF for event coreference:", EntityCEAF
