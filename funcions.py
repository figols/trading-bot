from contra import ALPHA
import pandas as pd
from collections import Counter
import requests, json, random, math, time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import operator as op
from numba import njit

def veles(dframe):
	dalt=[]
	baix=[]
	baj=[]
	for i in range(len(dframe)):
		if dframe.iloc[i]['Open']>=dframe.iloc[i]['Close']:
			dalt.append(dframe.iloc[i]['Open'])
			baix.append(dframe.iloc[i]['Close'])
			baj.append('bajista')
		else:
			dalt.append(dframe.iloc[i]['Close'])
			baix.append(dframe.iloc[i]['Open'])
			baj.append('alcista')

	dframe['M. sup']=dframe['High']-dalt
	dframe['Cos']=abs(dframe['Open']-dframe['Close'])
	dframe['M. inf']=baix-dframe['Low']
	dframe['Tipus']=baj

	pene=[]
	sup=dframe['M. sup'].mean()
	inf=dframe['M. inf'].mean()
	lok=dframe['Cos'].mean()
	coes=dframe['M. sup']+dframe['M. inf']
	koes=coes.mean()
	total=dframe['Cos']+dframe['M. sup']+dframe['M. inf']

	for i in range(len(dframe)):
		if dframe['Cos'][i]<=lok/12:
			if dframe['M. sup'][i]==0 and dframe['M. inf'][i]>0:
				pene.append('doji libelula')
			elif dframe['M. inf'][i]==0 and dframe['M. sup'][i]>0:
				pene.append('doji lapida')
			else:
				if coes[i]>=koes:
					pene.append('doji gran')
				else:
					pene.append('doji')
		elif coes[i]==0:
			pene.append('Marubozu')
		elif dframe['M. inf'][i]==0:
			pene.append('cul afaitat')
		elif dframe['M. sup'][i]==0:
			if coes[i]>=koes:
				pene.append('el penjat')
			else:
				pene.append('cap afaitat')
		elif dframe['Cos'][i]<=lok/2.5:
			if abs(dframe['M. sup'][i]-dframe['M. inf'][i])<=koes/2:
				pene.append('trompa')
			elif dframe['M. sup'][i]>=5*dframe['M. inf'][i]:
				pene.append('estel fugaç')
			else:
				if dframe['M. sup'][i]>dframe['M. inf'][i]:
					pene.append('martell invers')
				else:
					pene.append('martell')		
		elif dframe['M. sup'][i]<sup/10 and dframe['M. inf'][i]<inf/10:
			pene.append('Marubozu')
		elif dframe['M. sup'][i]>=2/3*total[i]:
			pene.append('gran ombra sup')
		elif dframe['M. inf'][i]>=2/3*total[i]:
			pene.append('gran ombra inf')
		elif dframe['Cos'][i]>=1.5*lok:
			pene.append('normal gran')
		else:
			pene.append('normal')
	dframe['Forma']=pene
	return dframe

def download(symbol,temp,vela):
	API_URL = "https://www.alphavantage.co/query"
 	#if symbol in symbols:
	if temp=='daily':
		temp="TIME_SERIES_DAILY"
	#elif temp=altres
	dades = { "function": temp, 
	"symbol": symbol,
	"outputsize" : "full",
	"datatype": "json", 
	"apikey": ALPHA }
	 

	response = requests.get(API_URL, dades).json()
	# pd.set_option('display.max_columns', None)
	data = pd.DataFrame.from_dict(response['Time Series (Daily)'], orient= 'index').sort_index(axis=1)
	data=data.iloc[::-1]
	data.reset_index(inplace=True)
	data = data.rename(columns={ 'index': 'Date','1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close',  '5. volume': 'Volume'})
	data=data.drop(columns='Volume')
	for i in range(len(data)):
		date=datetime.strptime(data['Date'].copy()[i],'%Y-%m-%d')
		data['Date'][i]=date.strftime('%d-%m-%Y')

	for column in data.columns[1::]:
		data[column]=pd.to_numeric(data[column])
	if vela==True:
		data=veles(data)
	return data

'varios indicadors'
def MA(tros,n):
	MA=[]
	for i in range(len(tros)):
		try:
			MA.append(round(sum(tros.iloc[i-n:i]['Close'])/n,3))
		except:
			MA.append(None)
	return MA

def RSI(tros):
	rsi=[]
	for j in range(len(tros)):
		difs=[round(tros.iloc[i+1]['Close']-tros.iloc[i]['Close'],2) for i in range(abs(j-14),j)]
		clos=[tros.iloc[i]['Close'] for i in range(abs(j-14),j)]
		# print(clos)
		if len(difs)==14:
			quoc=[round(100*difs[i]/clos[i],3) for i in range(len(difs))]
			# print(quoc)
			avgD=[term for term in quoc if term<0]
			avgD=abs(sum(avgD))/len(avgD)
			avgU=[term for term in quoc if term>=0]
			avgU=sum(avgU)/len(avgU)
			RS=avgU/avgD
			# print(menos)
			# print(mas)
			# print(sum(mas))
			# print(sum(menos))
			rsi.append(round(100-100/(1+RS),3))
		else:
			rsi.append(0)
		# print(j)
		# print(difs)
	# print(rsi)
	return rsi

def TR(tros):
	tr=[max([tros.iloc[i]['High']-tros.iloc[i]['Low'],abs(tros.iloc[i]['High']-tros.iloc[i]['Close']),abs(tros.iloc[i]['Low']-tros.iloc[i]['Close'])]) for i in range(len(tros))]
	tr=arredonir(tr,2)
	return tr

def ATR(tros):
	atr=[]
	tr=TR(tros)
	for i in range(len(tros)):
		try:
			atr.append(round(sum(tr[i-14:i])/14,3))
		except:
			atr.append(None)
	return atr

'tres tipus de veles'
def trenta(tros):
	trenta2=[tros.iloc[i]['Low']+0.382*(tros.iloc[i]['High']-tros.iloc[i]['Low']) for i in range(len(tros))]
	trenta3=[tros.iloc[i]['High']-0.382*(tros.iloc[i]['High']-tros.iloc[i]['Low']) for i in range(len(tros))]
	#print(trenta2[0:20],trenta3[0:20])
	trent=[]
	for i in range(len(tros)):
		if tros.iloc[i]['Open']>trenta3[i] and tros.iloc[i]['Close']>trenta3[i] and tros.iloc[i]['Tipus']=='alcista': 
			trent.append('dalt')
		elif tros.iloc[i]['Open']<trenta2[i] and tros.iloc[i]['Close']<trenta2[i] and tros.iloc[i]['Tipus']=='bajista':
			trent.append('baix')
		else:
			trent.append(False)
	return trent

def cierra(tros):
	tanca=[False]
	for i in range(1,len(tros)):
		if tros.iloc[i]['Close']<=tros.iloc[i-1]['Low']:
			tanca.append('baix')
		elif tros.iloc[i]['Close']>=tros.iloc[i-1]['High']:
			tanca.append('dalt')
		else:
			tanca.append(False)
	return tanca

def engulf(tros):
	engolf=[False]
	for i in range(1,len(tros)):
		if tros.iloc[i]['Cos']>=tros.iloc[i-1]['Cos']:
			if tros.iloc[i]['Tipus']=='alcista' and tros.iloc[i-1]['Tipus']=='bajista' and tros.iloc[i]['Close']>tros.iloc[i-1]['Open']:
				engolf.append('alcista')
			elif tros.iloc[i]['Tipus']=='bajista' and tros.iloc[i-1]['Tipus']=='alcista' and tros.iloc[i]['Open']>tros.iloc[i-1]['Close']:
				engolf.append('bajista')
			else: engolf.append(False)
		else: engolf.append(False)
	# print(len(engolf))
	return engolf

'funcionetes'
def extrems(dframe,marge,mins):
	super_max=[]
	super_max_pos=[]
	basic_super_max=[]

	i=0
	for num in dframe['High']:
		if i in range(marge,len(dframe['High'])-marge) and num==max(dframe.iloc[i-marge:i+marge+1]['High']):
			basic_super_max.append(num)
			super_max.append([i,num,'Max'])
			super_max_pos.append(i)
		i+=1
	super_min=[]
	super_min_pos=[]
	basic_super_min=[]
	i=0
	for num in dframe['Low']:
		if i in range(marge,len(dframe['Low'])-marge) and num==min(dframe.iloc[i-marge:i+marge+1]['Low']):
			basic_super_min.append(num)
			super_min.append([i,num,'Min'])
			super_min_pos.append(i)
		i+=1
	juan=basic_super_max+basic_super_min
	joint=super_max+super_min
	joint_pos=super_max_pos+super_min_pos
	aa=joint_pos.copy()
	joint_pos=sorted(joint_pos)
	ext=[]
	simple=[]
	for pos in aa:
		ee=aa.index(min(aa))
		ext.append(joint[ee])
		simple.append(juan[ee])
		aa[ee]=max(aa)+1
	if mins==False:
		return [ext,simple,joint_pos]
	if mins==True:
		return [ext,simple,joint_pos,super_max,super_min,super_max_pos,super_min_pos]

def tendencies(ext_llista):
	pos_mins=[] #posició de min en ext_llista
	pos_maxs=[] #posició de max en ext_llista
	for i in range(len(ext_llista)):
		if ext_llista[i][2]=='Max':
			pos_maxs.append(i)
		elif ext_llista[i][2]=='Min':
			pos_mins.append(i)

	#TENDENCIES
	eee=[]
	uuu=[]
	aaa=[]
	basic_min=[] #valors minims en conjunts ascendents
	tend_min=[] #[posicio en tros, valor, 'Min'] en conjunts ascendents
	pos_tend_min=[] #posicio en tros en conjunts ascendents
	orde=[term[2] for term in ext_llista] #seguit ordenat de 'Min' i 'Max'
	if len(pos_mins)!=0 and len(pos_mins)!=1:
		guardat=False
		pos_ult_min=len(orde)-1-orde[::-1].index('Min')
		for i, j in zip(pos_mins[0:-1], pos_mins[1:]): #minims ascendents
			if ext_llista[i][1]<=ext_llista[j][1] and guardat==False:
				eee.append(ext_llista[i])
				eee.append(ext_llista[j])
				uuu.append(ext_llista[i][1])
				uuu.append(ext_llista[j][1])
				aaa.append(ext_llista[i][0])
				aaa.append(ext_llista[j][0])
				guardat=True
			elif ext_llista[i][1]>ext_llista[j][1] and guardat==False:
				eee.append(ext_llista[i])
				uuu.append(ext_llista[i][1])
				aaa.append(ext_llista[i][0])
				tend_min.append(eee)
				basic_min.append(uuu)
				pos_tend_min.append(aaa)
				eee=[]
				uuu=[]
				aaa=[]
				eee.append(ext_llista[j])
				uuu.append(ext_llista[j][1])
				aaa.append(ext_llista[j][0])
				guardat=True
			elif ext_llista[i][1]<=ext_llista[j][1] and guardat==True:
				eee.append(ext_llista[j])
				uuu.append(ext_llista[j][1])
				aaa.append(ext_llista[j][0])
			else:
				tend_min.append(eee)
				basic_min.append(uuu)
				pos_tend_min.append(aaa)
				eee=[]
				uuu=[]
				aaa=[]
				eee.append(ext_llista[j])
				uuu.append(ext_llista[j][1])
				aaa.append(ext_llista[j][0])
			if j==pos_ult_min:
					tend_min.append(eee)
					basic_min.append(uuu)
					pos_tend_min.append(aaa)
	if len(pos_mins)==1:
		tend_min.append(ext_llista[pos_mins[0]])
		basic_min.append(ext_llista[pos_mins[0]][1])
		pos_tend_min.append(ext_llista[pos_mins[0]][0])
	eee=[]
	uuu=[]
	aaa=[]
	basic_max=[] #valors maxims en conjunts ascendents
	tend_max=[] #[posicio en tros, valor, 'Max'] en conjunts ascendents
	pos_tend_max=[] #posicio en tros en conjunts ascendents
	if len(pos_maxs)!=0 and len(pos_maxs)!=1:
		guardat=False
		pos_ult_max=len(orde)-1-orde[::-1].index('Max')
		for i, j in zip(pos_maxs[0:-1], pos_maxs[1:]): #maxims descendents
			if ext_llista[i][1]>=ext_llista[j][1] and guardat==False:
				eee.append(ext_llista[i])
				eee.append(ext_llista[j])
				uuu.append(ext_llista[i][1])
				uuu.append(ext_llista[j][1])
				aaa.append(ext_llista[i][0])
				aaa.append(ext_llista[j][0])
				guardat=True
			elif ext_llista[i][1]<ext_llista[j][1] and guardat==False:
				eee.append(ext_llista[i])
				uuu.append(ext_llista[i][1])
				aaa.append(ext_llista[i][0])
				tend_max.append(eee)
				basic_max.append(uuu)
				pos_tend_max.append(aaa)
				eee=[]
				uuu=[]
				aaa=[]
				eee.append(ext_llista[j])
				uuu.append(ext_llista[j][1])
				aaa.append(ext_llista[j][0])
				guardat=True
			elif ext_llista[i][1]>=ext_llista[j][1] and guardat==True:
				eee.append(ext_llista[j])
				uuu.append(ext_llista[j][1])
				aaa.append(ext_llista[j][0])
			else:
				tend_max.append(eee)
				basic_max.append(uuu)
				pos_tend_max.append(aaa)
				eee=[]
				uuu=[]
				aaa=[]
				eee.append(ext_llista[j])
				uuu.append(ext_llista[j][1])
				aaa.append(ext_llista[j][0])
			if j==pos_ult_max:
				tend_max.append(eee)
				basic_max.append(uuu)
				pos_tend_max.append(aaa)
	if len(pos_maxs)==1:
		tend_max.append(ext_llista[pos_maxs[0]])
		basic_max.append(ext_llista[pos_maxs[0]][1])
		pos_tend_max.append(ext_llista[pos_maxs[0]][0])
	return [tend_max,tend_min,pos_tend_max,pos_tend_min,basic_max,basic_min] 

def veïns(llista,pos_term):
	noo=[llista[i][2] for i in range(len(llista))]
	if pos_term!=-1:
		seg=llista[pos_term+1:]
		sego=noo[pos_term+1:]
	else:
		seg=[]
		sego=[]
	if pos_term!=0:
		ant=llista[pos_term-1::-1]
		anto=noo[pos_term-1::-1]
	else:
		ant=[]
		anto=[]
	if 'Min' not in sego:
		seg_min=None
	else:
		pos_seg_min=sego.index('Min')
		seg_min=seg[pos_seg_min]
	if 'Max' not in sego:
		seg_max=None
	else:
		pos_seg_max=sego.index('Max')
		seg_max=seg[pos_seg_max]
	if 'Min' not in anto:
		ant_min=None
	else:
		pos_ant_min=anto.index('Min')
		ant_min=ant[pos_ant_min]
	if 'Max' not in anto:
		ant_max=None
	else:
		pos_ant_max=anto.index('Max')
		ant_max=ant[pos_ant_max]
	return [ant_max,ant_min,seg_max,seg_min] 

def min_quad(x,y):
	if iguals(y):
		return [0,y[0],0,1,'ascendent']
	X = np.array(x)
	Y = np.array(y)
	A = np.vstack([X, np.ones(len(x))]).T
	m, c = np.linalg.lstsq(A, Y, rcond=None)[0]
	angle=np.arctan(m)*180/np.pi
	r2=np.corrcoef(X,Y)[0,1]**2
	if m>=0:
		asc='ascendent'
	else:
		asc='descendent'
	if len(X)>1:
		vector=[round(m, 2),round(c, 2),round(angle, 2),round(r2, 2),asc]
		#pendent, ord. origen, angle, r2, asc o desc
	else:
		vector=y
	return vector 

def tamany(dframe):
	ola=extrems(dframe,1,False)[2]
	lens=[len(extrems(dframe,i,False)[2]) for i in range(1,len(ola))]
	#nº de minmax per a marges ascendents
	# print('lens',lens)
	mylist=list(dict.fromkeys(lens))
	#nº de minmax per a marges amb diferent nº de minmaxs
	# print('mylist',mylist)
	if 0 in mylist:
		mylist.remove(0)
	# print('mylist',mylist)
	xee=[[mylist[i],lens.index(mylist[i])+1] for i in range(len(mylist))]
	#[nº de maxmin,marge més menut per a eixe nº]
	# print('xee',xee)
	marges=[xee[i][1] for i in range(len(xee))] #segon terme de xee
	# print('marges',marges)
	tot=[extrems(dframe,marge,False)[0] for marge in marges]
	# print('tot',tot)
	if not tot:
		return [[],[],[],[],[]]
	#calcular tots els extrems per als diferents marges
	new=tot[0].copy()
	xaa=[xee[i][1] for i in range(len(xee))]
	# print('xaa',xaa)
	j=0
	for i in range(len(tot)):
		for term in tot[i]:
			if i<len(tot)-1:
				if term not in tot[i+1] and term!=[]:
					new[new.index(term)].append(xaa[j])	
			elif len(term)==3:
				new[new.index(term)].append(xaa[j])
		j+=1
	tot = list(filter(None, tot))
	# print('tot',tot)
	for _ in range(len(lens)):
		if 0 in lens:
			lens.remove(0)
	return [new,Counter(lens),tot,xee,marges]

def termes(llista,term):
	liste=llista.copy()
	pos=[]
	ese=liste.count(term)
	for _ in range(ese):
		lloc=liste.index(term)
		pos.append(lloc)
		liste[lloc]=term+1
	# print(liste)
	return pos

def list_in(a,b):
	return any(map(lambda x: b[x:x+len(a)]==a,range(len(b)-len(a)+1)))

def iguals(llista):
	return all(term==llista[0] for term in llista)

def asc(llista):
	return all(map(lambda x: llista[x]+1==llista[x+1],range(len(llista)-1)))

def arredonir(llista,num):
	return list(map(lambda x: round(x,num),llista))

def aplanar(llista):
	return [term for termo in llista for term in termo]

def ordo(llista):
	def ord(e):
		return e[-1]
	for term in llista: term.append(abs(term[5]-term[6]))
	llista.sort(key=ord,reverse=True)
	for term in llista: del term[-1]
	return llista

def estructura(dframe,trosses,num,prec):
	mostra=list(dframe['Mig'][trosses[num][0][0]:trosses[num][-1][0]+1])
	poses=list(range(trosses[num][0][0],trosses[num][-1][0]+1))
	# print(poses)
	grande=[]
	grandissim=[]
	grando=[]
	grandissimo=[]
	# granet=[]
	for llarg in range(2,len(mostra)+1):
		mit=[]
		lit=[]
		sit=[]
		for i in range(0,len(mostra)+1-llarg):
			quatros=min_quad(poses[i:i+llarg],mostra[i:i+llarg])
			quatros.append(poses[i])
			quatros.append(poses[i+llarg-1])
			mit.append(quatros)
			# print('i',i)
			if quatros[3]>=prec:
				lit.append(quatros)
				# if abs(quatros[5]-quatros[6])>1:
				# 	sit.append(quatros)
		grande.append([llarg,mit])
		grandissim.append(mit)
		grando.append([llarg,lit])
		grandissimo.append(lit)
		# print('llarg',llarg)
	# print('grande:',grande)
	# print('grandissim:',grandissim)
	# print('grando:',grando)
	# print('grandissimo:',grandissimo)

	planet=[term for termo in grandissimo for term in termo]
	# print('planet',planet)
	total=[planet[-1]]
	# print('total',total)
	a=0
	# print(poses)
	while total[0][5]!=poses[0] or total[-1][6]!=poses[-1]:
		aux_inici=[]
		aux_final=[]
		for term in planet:
			if term[6]==total[0][5]:
				aux_inici.append(term)
			elif term[5]==total[-1][6]:
				aux_final.append(term)
		# print('aux_inici',aux_inici)
		# print('aux_final',aux_final)
		if aux_inici:
			total.insert(0,ordo(aux_inici)[0])
		if aux_final:
			total.append(ordo(aux_final)[0])
		a+=1
	# print(total)
	return total

def tends_complet(nou,marges,printo,purga):
	if not nou:
		return [[],[],[],[],[]]
	tot_min_quads=[]
	tot_min_quats=[]
	sak=[]
	trosses=[]
	molt=0
	esee=[0,0,0,0,0,0,0,0,0,0,0,0,0]
	i=1
	while asc(marges[0:i]) and i<len(marges)+1:
		nigus=marges[i-1:] #marges amb dif major a 1
		i+=1
	if len(nigus)==1: #canviar si pocs max/min
		nigus=marges[math.ceil(len(marges)/2)+suaj-1:] #la meitat si no suficients
	# print('marges',marges)
	# print('nigus',nigus)
	# print('nou:',nou)
	principi=True
	while len(esee)>3:
		mal=nigus[molt:]
		# print('mal:',mal)
		esee=[term for term in nou if term[3] in mal] #mins i maxs més grans en extrems
		# print('esee:',esee)
		eixe=[nou.index(term) for term in esee] #posicions d'estos mins/maxs 
		# en llista d'extrems
		# print('eixe1:',eixe)
		primer=nou[0:eixe[0]]
		# print('primer:',primer)
		prim_valors=[term[1] for term in primer] # valors d'accions per a pos en primer
		# print('primer_valors:',prim_valors)
		if principi==True:
			eixe_ant=eixe
		prim_min=primer[prim_valors.index(min(prim_valors))]
		prim_max=primer[prim_valors.index(max(prim_valors))]
		if principi==False:
			cond_min=nou.index(prim_min) in eixe_ant
			cond_max=nou.index(prim_max) in eixe_ant
		else:
			cond_min=True
			cond_max=True
		if esee[0][2]=='Max' and cond_min:
			esee.insert(0,prim_min)
		elif esee[0][2]=='Min' and cond_max:
			esee.insert(0,prim_max)
		ultim=nou[eixe[-1]+1:]
		# print('ultim:',ultim)
		ultim_valors=[term[1] for term in ultim]
		# print('ultim_valors:',ultim_valors)
		ult_min=ultim[ultim_valors.index(min(ultim_valors))]
		ult_max=ultim[ultim_valors.index(max(ultim_valors))]
		if principi==False:
			cond_min=nou.index(ult_min) in eixe_ant
			cond_max=nou.index(ult_max) in eixe_ant
		else:
			cond_min=True
			cond_max=True
		if esee[-1][2]=='Max' and cond_min:
			esee.append(ult_min)
			# print('max',ultim_valors.index(min(ultim_valors)))
			# print('aberració:',nou.index(ultim[ultim_valors.index(min(ultim_valors))]))
		elif esee[-1][2]=='Min' and cond_max:
			esee.append(ult_max)
			# print('min',ultim_valors.index(max(ultim_valors)))
		# print(esee)
		eixe=[nou.index(term) for term in esee]
		# print('eixe2:',eixe)
		if principi==True:
			principi=False
			eixe_ant=eixe
		trossos=[nou[i:j+1] for i, j
		in zip(eixe[0:-1],eixe[1:])] #seguit de maxs/mins entre cada min/max gran
		# print('trossos',trossos)
		sok=[tendencies(tross) for tross in trossos] #tendencies a partir dels trossos
		sup_base=[]
		sup_pose=[]
		base=[]
		pose=[]
		j=0
		for term in trossos:
			for i in range(len(term)):
				base.append(term[i][1])
				pose.append(term[i][0])
			sup_base.append(base)
			sup_pose.append(pose)
			base=[]
			pose=[]
			if term not in trosses:
				trosses.append(term)
				sak.append(sok[j])
			j+=1
		# print(sup_pose,sup_base)
		min_quads=[]
		min_quats=[]
		for i in range(len(sup_base)):
			ye=min_quad(sup_pose[i],sup_base[i])
			ye1=ye.copy()
			ye1.append(molt)
			min_quats.append(ye)
			min_quads.append(ye1)
			if ye not in tot_min_quats:
				tot_min_quats.append(ye)
				tot_min_quads.append(ye1)
		molt+=1
		eso=[term[3] for term in esee]
		if iguals(eso):
			break
	# for i in range(len(trosses)):
	# 	print(f'tros {i}: {trosses[i]}')
	macho=[]
	for term in trosses:
		nomes=[]
		for cosa in trosses:
			if list_in(cosa,term):
				nomes.append(trosses.index(cosa))
		macho.append(nomes)

	# for i in range(len(trossos)):
	# 	print(f'tros {i}: {trossos[i]}')
	mache=[term for term in macho if len(term)==1]
	mache=[item for sublist in mache for item in sublist]
	for term in macho:
		termo=term.copy()
		for pos in term:
			# print(pos)
			if pos not in mache:
				termo.remove(pos)
			# print(term) 
		macho[macho.index(term)]=termo
	# print('macho:',macho)
	# print('tot_min_quads:',tot_min_quads)
	# print('tot_min_quats:',tot_min_quats)
	# for term in sak:
	# 	print('term:',term)
	# print('trosses:',trosses)
	borra=[]
	if purga==True:
		# print(len(macho),len(tot_min_quads),len(tot_min_quats),
		# 	len(sak),len(trosses))
		rang=range(len(sak))
		# print(rang)
		for pos in rang:
			if len(trosses[pos])==2:
				borra.append(pos)
			if tot_min_quads[macho[pos][0]][4]!=tot_min_quads[macho[pos][-1]][4]:
				borra.append(pos)
		# print(borra)
		rang=[term for term in rang if term not in borra]
		# print(rang)
		
		# for pos in borra:
		# 	del macho[pos]
		# 	del tot_min_quads[pos]
		# 	del tot_min_quats[pos]
		# 	del sak[pos]
		# 	del trosses[pos]
		# print(len(macho),len(tot_min_quads),len(tot_min_quats),
		# 	len(sak),len(trosses),len(trossos))
		for pos in rang:
			princip=[term[1] for term in trosses[macho[pos][0]]]
			fine=[term[1] for term in trosses[macho[pos][-1]]]
			# print('pos:',pos)
			# print(fine)
			# print(princip)
			if tot_min_quads[macho[pos][0]][4]=='ascendent' and min(princip)>min(fine):
				borra.append(pos)
			if tot_min_quads[macho[pos][0]][4]=='descendent' and max(princip)<max(fine):
				borra.append(pos)
		# print(borra)
		# borra=list(reversed(sorted(borra)))

		# for pos in borra:
		# 	del macho[pos]
		# 	del tot_min_quads[pos]
		# 	del tot_min_quats[pos]
		# 	del sak[pos]
		# 	del trosses[pos]
	if printo==True:
		for i in range(len(sak)):
			u=i+1
			print(f'Figura {u}')
			if i in borra:
				print('CANSELADO')
			print(f'conté {macho[i]}')
			print(tot_min_quads[i])
			print('tendències:')
			print(sak[i][0])
			print(sak[i][1])
			print('màxims i mínims:')
			print(f'de {trosses[i][0]} a {trosses[i][-1]}')
	return [tot_min_quads,sak,trosses,macho,mache,borra]

def ajust(part,pendent,ordenada):
	dif=[] #diferencia entre cada valor i punt de la recta
	baix=[] #diferencies per baix 
	dalt=[] #diferencies per dalt
	for i in range(len(part)):
		d=round(part[i]-pendent*i-ordenada, 6)
		dif.append(d)
		if d>=0:
			dalt.append(d)
		else:
			baix.append(d)
	suma=round(sum(dif),6) #suma de diferencies entre
	#cada valor i punt de la recta
	sum_dalt=round(sum(dalt), 6) #suma dels de dalt
	sum_baix=round(sum(baix), 6) #suma dels de baix
	return [suma,sum_dalt,sum_baix]

def compara(a,b,arg):
	if arg==True:
		return op.le(a,b)
	else:
		return op.ge(a,b)

def busca(llista): # troba els asc o desc duplicats seguits
	rek=[]
	if llista[0]==llista[1]:
		rek.append(0)
	for num in range(1,len(llista)-1):
		if llista[num-1]==llista[num] or llista[num]==llista[num+1]:
			rek.append(num)
	if llista[-1]==llista[-2]:
		rek.append(len(llista)-1)
	return rek

def supera(a,b,asck):
	if asck==True:
		return op.gt(a,b)
	else:
		return op.lt(a,b)

def dolor(mak,asck): #tendències asc (min_n+1>min_n, max_n+1>max_n) o desc en mak
	pain=[] # sèries tal que term[2][i]> term[1][i-1] o term[2][i]< term[1][i-1]
	for pos in range(len(mak)-1):
		ask=[mak[pos]]
		for i in range(len(mak)-pos):
			if pos+i<len(mak)-1:
				if supera(mak[pos+i+1][2],mak[pos+i][1],asck):
					ask.append(mak[pos+i+1])
				else:
					if len(ask)>1:
						# print('ask',ask)
						pain.append(ask)
					break
			else:
				if len(ask)>1:
					# print('ask',ask)
					pain.append(ask)
				break
	# pain=[term for term in pain if len(term)%2==1] #neteja número parells
	# print('pain',pain)

	for _ in range(len(pain)): #neteja de termes continguts en altres termes
		merda=[] #llista de [a, b en a]
		kedise=[] #llista de [len(a),len(b)]
		# print('pain',pain)
		for term in pain:
			for este in pain:
				pal=list_in(term,este)
				if pal:
					uos=[pain.index(term),pain.index(este)] #[a, b en a]
					if uos[0]!=uos[1]:
						merda.append(uos)
						# print(uos)
		for term in merda:
			# print('merda term',term)
			posk=[len(pain[term[0]]),len(pain[term[1]])] # [len(a),len(b)]
			kedise.append(posk)
			# print('posk',posk)
		if not kedise:
			break
		# print('kedise',kedise)
		esto=kedise[0].index(min(kedise[0])) #quin nombre és menor del 1r terme de kedise
		# print('esto',esto)
		# print(merda[0][esto])
		del pain[merda[0][esto]] #eliminem aquest terme
		# print('pain',pain)

	if asck==True:
		for term in pain:
			if term[-1][0]=='descendent':
				del term[-1]
			if term[0][0]=='descendent':
				del term[0]
		pain=[term for term in pain if len(term)>1]
	else:
		for term in pain:
			if term[-1][0]=='ascendent':
				del term[-1]
			if term[0][0]=='ascendent':
				del term[0]
		pain=[term for term in pain if len(term)>1]
		# print('pain',pain)
	return pain

def neteja(planet,tros): #canvia els mins quad més curts seguits per un min quad conjunt
	longs=[]
	for term in planet:#tots els ajustos amb r2>.95 [pend,ord. origen,angle,r2,asc/desc]
		maa=math.sqrt((term[6]-term[5])**2+(tros['Mig'][term[6]]-tros['Mig'][term[5]])**2)
		longs.append(round(maa,3)) #longitud entre 1r i ultim punt de cada ajust de planet
	norm=[round(term,2) for term in np.divide(longs,max(longs))]
	#longs normalitzats a max(longs)
	# print('norm',norm)
	# ordenat=list(reversed(sorted(list(dict.fromkeys(norm)))))
	
	nums=[i for i in range(len(norm)) if norm[i]<0.1]
	#pos de termes amb norm<0.1 en planet (no en tros)
	# print('nums',nums)
	tru=[] #True si dos nombres seguits [4,5], False si no [4,8], [9,12]
	for i in range(len(nums[0:-1])):
		if nums[i+1]-nums[i]==1:
			tru.append(True)
		else:
			tru.append(False)
	tru.append(False)
	# print('tru',tru)

	misa=[] #cadenes de nombres consecutius en nums [1,2,3,5,6,8]->[[1,2,3],[5,6],[8]]
	ara=[]
	for i in range(len(nums)):
		if not ara:
			ara=[nums[i]]
		if tru[i]==False:
			if nums[i] not in ara:
				ara.append(nums[i])
			misa.append(ara)
			ara=[]
		elif tru[i]==True and nums[i] not in ara:
			ara.append(nums[i])
	# print('misa',misa)
	noves=[[planet[term[0]][5],planet[term[-1]][6]] for term in misa]
	# posició en tros de primer i últim terme dels termes de misa
	# print('noves',noves)
	# print('planet',planet)

	plants=[min_quad(range(term[0],term[1]+1),list(tros['Mig'].iloc[range(term[0],term[1]+1)]))+term for term in noves]
	# mínims quadrats entre les posicions en noves

	for i in range(len(misa)): 
		planet[misa[i][0]]=plants[i]
		for j in misa[i][1:]:
			planet[j]=j

	#substituïm els min quads curts pels nous min quads en planet
	for i in reversed(range(len(planet))): 
		if type(planet[i])==int:
			del planet[i]
	#planet definitiu
	llox=[[term[5],term[6]] for term in planet] #pos inicial i final de cada pendent
	# print(llox)
	mek=[] #['asc' o 'desc', primer 'Low'/'High', últim 'High'/'Low'] de cada pendent
	for term in planet:
		if term[4]=='ascendent':
			mek.append([term[4],tros['Low'][term[5]],tros['High'][term[6]]])
		else:
			mek.append([term[4],tros['High'][term[5]],tros['Low'][term[6]]])
	# print('mek',mek) 
	puj=[term[0] for term in mek] # llista de asc/desc
	for _ in range(len(puj)):
	# neteja els asc o desc duplicats seguits i termes corresponents en mek i llox
		seep=busca(puj)
		# print('seep',seep)
		if seep:
			mek[seep[0]]=[puj[seep[0]],mek[seep[0]][1],mek[seep[1]][2]]
			llox[seep[0]]=[llox[seep[0]][0],llox[seep[1]][1]]
			del mek[seep[1]]
			del puj[seep[1]]
			del llox[seep[1]]
		else:
			break
		# print('mek',mek)
		# print('puj',puj)
	# print('mek',mek)
	# print('puj',puj)
	# print('llox',llox)
	mak=[mek[i]+llox[i] for i in range(len(mek))]
	#junta termes seguits asc o desc de planet
	#['asc', primer 'Low', últim 'High', pos 1r en tros, pos ult en tros] de les pendents
	print('mak',mak)
	pain=dolor(mak,True) #tends ascendents de mak (min_n+1>min_n, max_n+1>max_n)
	pain+=dolor(mak,False) #tends descendents de mak (min_n+1<min_n, max_n+1<max_n)
	for term in pain: #ordenant per ordre 
		term[0].insert(0,term[0][3])
	pain=list(sorted(pain))
	for term in pain:
		del term[0][0]	
	return [planet,pain]

def percentils(nuevo,a): # pren una llista de nombres i la divideix en intervals 
	# de tres en tres del valor més menut al més gran
	oe=list(range(round(min(nuevo)/10)*10-6-a,round(max(nuevo)/10)*10+6+a,3))
	novet=nuevo.copy()
	i=len(oe)-1
	for term in reversed(oe): #cada min/max en quin percentil es trobe
		temp=list(np.array(novet)-term) #reste a cada nre de nou cada valor de oe
		for j in range(len(nuevo)):
			if temp[j]>=0 and type(nuevo[j])==float: 
				# print(type(nuevo[j]))
				nuevo[j]=[nuevo[j],i]
		i-=1
	return [nuevo,oe]

def limpia(ultra,poses): #[1,1,1,1,1],[1,1,1,1,1] seguits -> [1,1,1,1,1,1]
	for _ in range(len(ultra)):
		yes=True
		for j in range(len(ultra)):
			for i in reversed(range(len(ultra)+1)):
				# print('[j,i]',[j,i])
				if iguals(ultra[j:i]) and len(ultra[j:i])>1 and set(poses[j]).intersection(*poses[j:i]):
					# print('poses',poses)
					# print('ultra',ultra)
					# print('delet',ultra[j:i])
					# print('delet',poses[j:i])
					del ultra[j:i]
					del poses[j:i]
					yes=False
				if 	yes==False:
					break	
			if 	yes==False:
					break
		if yes==True:
			break
	return [ultra,poses]

def suport(tros,nou,maks): #nivells horitzontals de suport/resistència
	# maks='Max'
	if maks=='Max':
		jai='High'
		maj=False
	elif maks=='Min':
		jai='Low'
		maj=True
	Mega=[]
	Giga=[]
	Ea=[]
	Cond=[]
	for r in [0,1,2]:
		nuevo1=[term[1] for term in nou if term[2]==maks] #valor del max/min
		# print('nuevo1',nuevo1)
		nop1=[term[0] for term in nou if term[2]==maks] #pos. en tros
		# print('nop1',nop1)
		[nuevo1,oe]=percentils(nuevo1,r)
		# print('nuevo1',nuevo1)
		# print('oe',oe) #valors dels intervals
		percs1=[term[1] for term in nuevo1] #en quin percentil es trobe cada max
		# print('percs1',percs1)
		# print('counter percs1',Counter(percs1))
		mega=[]
		giga=[]
		for j in range(5,11): #de 5 a 11 màxims seguits
			ultra=[percs1[a:a+j] for a in range(0,len(percs1)-j+1) if iguals(percs1[a:a+j])]
			poses=[nop1[a:a+j] for a in range(0,len(percs1)-j+1) if iguals(percs1[a:a+j])]
			[ultra,poses]=limpia(ultra,poses) #maxs seguits en mateix perctl i pos
			# if ultra and poses:
			# 	print('ultra',ultra)
			# 	print('poses',poses)
			mega.append(ultra) #maxs
			giga.append(poses) #pos
		giga=aplanar(giga) #lleve elements nuls
		mega=aplanar(mega) #lleve elements nuls
		# print('giga',giga)
		# print('mega',mega)
		Giga.append(giga)
		Mega.append(mega)
		# print('Giga',Giga)
		# print('Mega',Mega)
		ea=[oe[term[0]] for term in mega] #limits inf de valor de maxs seguits (interval [ea,ea+3]) 
		Ea.append(ea)
		# print('oe',oe)
		# print('ea',ea)
		condicio=[]
		for term in giga:
			lang=term[-1]-term[0] #resta entre ultima i 1a pos
			abans=term[0]-lang #marge inferior de tamany lang
			desp=term[-1]+lang #marge superior de  "  "  "
			ante=list(tros.iloc[abans:term[0]][jai]) #termes 'High'/'Low' del marge inf
			termo=list(tros.iloc[term[0]:term[-1]+1][jai]) #termes 'High'/'Low' del marge de dins
			poste=list(tros.iloc[term[-1]:desp][jai]) #termes 'High'/'Low' del marge sup
			# print('abans',abans)
			# print(term[0])
			# print(term[-1])
			# print('desp',desp)
			# print('ante',ante)
			# print('mitja',np.mean(ante))
			# print(f'tros.iloc[{term[0]}][{jai}]',tros.iloc[term[0]][jai])
			# print('termo',termo)
			# print('mitja',arredonir([np.mean(ante),np.mean(termo),np.mean(poste)],2))
			# print(f'tros.iloc[{term[-1]}][{jai}]',tros.iloc[term[-1]][jai])
			# print('poste',poste)
			# print('mitja',np.mean(poste))
			condicio.append(supera(np.mean(poste),np.mean(termo),maj) and supera(np.mean(ante),np.mean(termo),maj))
			# compare la mitjana de poste/ante amb mitjana de termo (maj <,!maj >)
			# si els dos marges són majors/menors done True
		Cond.append(condicio)
		# print('condicio',condicio)


	# print('Giga',Giga)
	# print('Mega',Mega)
	# print('Ea',Ea)
	# print('Cond',Cond)
	for i in range(len(Ea)):
		Giga[i]=[Giga[i][j] for j in range(len(Giga[i])) if Cond[i][j]==True]
		Mega[i]=[Mega[i][j] for j in range(len(Mega[i])) if Cond[i][j]==True]
		Ea[i]=[Ea[i][j] for j in range(len(Ea[i])) if Cond[i][j]==True]
	# print('Giga',Giga)
	# print('Mega',Mega)
	# print('Ea',Ea)
	# print('Cond',Cond)
	return [Giga,Mega,Ea]

def analesi(lista,printo): #estadístics ML --
	listo=(np.array(lista)-min(lista))/max(lista) #normalitzar
	mediana=round(np.median(listo),4) #mediana
	mitja=round(np.mean(listo),4) #mitja
	recta=abs(plans(lista)[0]) #pendent de plans
	pos=range(len(lista))
	pendent=min_quad(pos,lista) #pendent de min quadrats
	r2=pendent[3]
	pendent=abs(round(pendent[0],4))
	paxos=[] #posicions de maxs amb el marge corresponent
	pinos=[] #posicions de mins amb el marge corresponent
	for marge in range(1,int(np.ceil(len(lista)/2))):
		i=0
		pos_max=[]
		maxs=[]
		pos_min=[]
		mins=[]
		super_max=[]
		super_min=[]
		for num in lista:
			if i in range(marge,len(lista)-marge) and num==max(lista[i-marge:i+marge+1]):
				maxs.append(num)
				pos_max.append(i)
				super_max.append([i,num,'Max'])
			if i in range(marge,len(lista)-marge) and num==min(lista[i-marge:i+marge+1]):
				mins.append(num)
				pos_min.append(i)
				super_min.append([i,num,'Min'])
			i+=1
			juan=maxs+mins
			joint=super_max+super_min
			joint_pos=pos_max+pos_min
			aa=joint_pos.copy()
			joint_pos=sorted(joint_pos)
			ext=[]
			simple=[]
			for pos in aa:
				uu=aa.index(min(aa))
				ext.append(joint[uu])
				simple.append(juan[uu])
				aa[uu]=max(aa)+1
		paxos.append([marge,pos_max])
		pinos.append([marge,pos_min])

		# print('pos_max',pos_max)
		# print('maxs',maxs)
		# print('pos_min',pos_min)
		# print('mins',mins)
		# print('joint_pos',joint_pos)
		# print('juan',juan)
		# print('joint',joint)
		# print('uu',uu)
		# print('marge',marge)
		# print('ext',ext)

	# print('paxos',paxos)
	# print('pinos',pinos)

	for i in range(len(paxos)-1):
		for j in reversed(range(len(paxos[i][1]))):
			if paxos[i][1][j] in paxos[i+1][1]:
				del paxos[i][1][j]
	for i in range(len(pinos)-1):
		for j in reversed(range(len(pinos[i][1]))):
			if pinos[i][1][j] in pinos[i+1][1]:
				del pinos[i][1][j]

	sumeta_max=[len(term[1]) for term in paxos] #nombre de maxs per cada marge
	tot_max=round(sum(sumeta_max)/len(lista),4) #percentatge de valors que son maxs
	sumeta_min=[len(term[1]) for term in pinos] #nombre de mins per cada marge
	tot_min=round(sum(sumeta_min)/len(lista),4) #percentatge de valors que son mins
	try:
		micha_max=round(sum([term[0]*len(term[1]) for term in paxos])/sum(sumeta_max),4) #marge mig de max
	except ZeroDivisionError:
		micha_max=0
	try:
		micha_min=round(sum([term[0]*len(term[1]) for term in pinos])/sum(sumeta_min),4) #marge mig de min
	except ZeroDivisionError:
		micha_min=0
	if printo:
		print('len(lista)',len(lista))
		print('llista',lista)
		print('llista ordenada',sorted(lista))
		print('mediana',mediana)
		print('mitja',mitja)
		print('recta plans',recta)
		print('pendent min quad',pendent)
		print('posicions de maxs amb el marge corresponent',paxos)
		print('nombre de max per cada marge',sumeta_max)
		print('sum(sumeta_max)',sum(sumeta_max))
		print('percentatge de valors que son max',tot_max)
		print('marge mig de max',micha_max)
		print('posicions de mins amb el marge corresponent',pinos)
		print('nombre de mins per cada marge',sumeta_min)
		print('sum(sumeta_min)',sum(sumeta_min))
		print('percentatge de valors que son mins',tot_min)
		print('marge mig de min',micha_min)
		print('r2',r2)
		print('')
	return [mediana,mitja,recta,pendent,tot_max,tot_min,micha_max,micha_min,r2] 

def funcioneta(part,list_ord): #per a plans --
	menor=[abs(ajust(part,n,list_ord)[0]) for n in [0.01,-0.01]]
	# print(menor)
	signe=menor.index(min(menor))
	signe=(-1)**signe

	list_pend=arredonir(list(signe*np.array([1,2,3,4])),5)
	# print(list_pend)
	distos=[]
	for i in list_pend:
		ajusto=ajust(part,i,list_ord)
		distos.append(abs(ajusto[0]))
	# print('distos',distos)
	pendos=list_pend[distos.index(min(distos))]
	# print(pendos)

	list_pend=arredonir(list(np.arange(pendos-1,pendos+1.5,0.5)),4)
	# print(list_pend)
	distos=[]
	for i in list_pend:
		ajusto=ajust(part,i,list_ord)
		distos.append(abs(ajusto[0]))
	# print('distos',distos)
	pendos=list_pend[distos.index(min(distos))]
	# print(pendos)

	list_pend=arredonir(list(np.arange(pendos-0.5,pendos+0.55,0.05)),4)
	# print(list_pend)
	distos=[]
	for i in list_pend:
		ajusto=ajust(part,i,list_ord)
		distos.append(abs(ajusto[0]))
	# print('distos',distos)
	pendos=list_pend[distos.index(min(distos))]
	# print(pendos)

	list_pend=arredonir(list(np.arange(pendos-0.05,pendos+0.055,0.005)),5)
	# print(list_pend)
	distos=[]
	for i in list_pend:
		ajusto=ajust(part,i,list_ord)
		distos.append(abs(ajusto[0]))
	# print('distos',distos)
	pendos=list_pend[distos.index(min(distos))]
	# print(pendos)

	list_pend=arredonir(list(np.arange(pendos-0.005,pendos+0.0055,0.0005)),5)
	# print(list_pend)
	distos=[]
	for i in list_pend:
		ajusto=ajust(part,i,list_ord)
		distos.append(abs(ajusto[0]))
	# print('distos',distos)
	pendos=list_pend[distos.index(min(distos))]
	# print(pendos)

	list_pend=arredonir(list(np.arange(pendos-0.0005,pendos+0.00051,0.0001)),6)
	# print(list_pend)
	distos=[]
	saltos=[]
	for i in list_pend:
		ajusto=ajust(part,i,list_ord)
		distos.append(abs(ajusto[0]))
		saltos.append(abs(ajusto[1])+abs(ajusto[2]))
	# print('distos',distos)
	# print('saltos',saltos)
	pendos=list_pend[distos.index(min(distos))]
	saltros=saltos[distos.index(min(distos))]
	# distros=distos[distos.index(min(distos))]
	# print(saltros)
	# print(pendos)
	return [pendos,saltros]

def plans(part): # similar a min quad --
	ordos=range(round(np.floor(min(part))),round(np.ceil(max(part))))
	# print('ordos',ordos)
	pend=[]
	dif=[]
	for j in ordos:
		ajust=funcioneta(part,j)
		pend.append(ajust[0])
		dif.append(ajust[1])
	# print('pend',pend)
	# print('dif',dif)
	ordr=ordos[dif.index(min(dif))]
	# print('ordr',ordr)

	ordos=np.arange(ordr-3,ordr+3.1,0.1)
	# print('ordos',ordos)
	pend=[]
	dif=[]
	for j in ordos:
		ajust=funcioneta(part,j)
		pend.append(ajust[0])
		dif.append(ajust[1])
	# print('pend',pend)
	# print('dif',arredonir(dif,6))
	poses=[i for i in range(len(dif)) if dif[i]==min(dif)]
	# print('poses',poses)

	# ordos=np.arange(ordos[poses[0]]-0.5,ordos[poses[-1]]+0.51,0.01)
	# # print('ordos',ordos)
	# pend=[]
	# dif=[]
	# for j in ordos:
	# 	ajust=funcioneta(part,j)
	# 	pend.append(ajust[0])
	# 	dif.append(ajust[1])
	# # print('pend',pend)
	# # print('dif',arredonir(dif,6))
	# poses=[i for i in range(len(dif)) if dif[i]==min(dif)]
	# # print('poses',poses)

	pendets=[pend[i] for i in poses]
	ordets=[round(ordos[i],5) for i in poses]
	
	# print('pendets',pendets)
	# print('ordets',ordets)

	pan=round(sum(pendets)/len(pendets),5)
	ond=round(sum(ordets)/len(ordets),5)
	angle=np.arctan(pan)*180/np.pi
	if pan>=0:
		asc='ascendent'
	else:
		asc='descendent'
	return [pan,ond,round(angle, 2),0,asc]

def plans_mal(part): #plans més lent --
	reond=arredonir(part,1)
	if iguals(reond):
		return [0,part[0]]
	part=arredonir(part,5)
	ordenada=min(part) #ordenada inicial
	comparo=max(part)
	for vari in [1,0.1,0.01,0.001]:
		pendos=[] #pendents tal que saltos[i]=daxos[i]
		ordos=[] #ordenades tal que saltos[i]=daxos[i]
		saltos=[] #ultima suma de diferences entre recta i valors per damunt
		daxos=[] #ultima suma de diferences entre recta i valors per davall
		exe=0
		while ordenada<comparo:
			pendent=0
			sumas=[]
			pends=[] #pendents
			daltos=[] #diferencies entre recta i valors per damunt
			baixos=[] #diferencies entre recta i valors per davall
			var=0.1
			suma=1
			prim=True
			temps=time.time()
			while abs(suma)>0.01:
				suma=(-1)**exe
				ult_pend=[]
				while compara(suma,0,exe%2):
					if not prim:
						pendent=pendent+(-1)**exe*var
					prim=False
					[suma,sum_dalt,sum_baix]=ajust(part,pendent,ordenada)
					pends.append(round(pendent,12))
					ult_pend.append(pendent)
				if len(ult_pend)==1:
					pendent=ult_pend[-1]
				else:
					pendent=ult_pend[-2]
				sumas.append(round(suma, 3))
				var=var/10
				daltos.append(sum_dalt)
				baixos.append(sum_baix)
				prim=True
				if len(sumas)>20:
					# print('KKKANVI')
					break
			if iguals(pends):
				exe+=1
			else:
				# print(f'ORDENADA: {round(ordenada,5)}')
				# print(f'SUMAS: {sumas}')
				ordos.append(round(ordenada,7))
				saltos.append(daltos[-1])
				daxos.append(baixos[-1])
				ee=math.floor(-math.log(var/100,10))
				pends=list(map(lambda x: round(x,ee),pends))
				# print(f'PENDS: {pends}')
				pendos.append(pends[-1])
				ordenada+=vari
		# print('RESULTATS:')
		# print('suma superiors:',saltos)
		# print('suma inferiors:',daxos)
		# print('ordenades:',ordos)
		# print('pendents:',pendos)
		# print(iguals([len(saltos),len(daxos),len(ordos),len(pendos)]))
		# print(min(saltos))
		poss=saltos.index(min(saltos))
		# print('RECTA FINAL:')
		# print(f'{pendos[poss]}x+{ordos[poss]}')
		# print(pendos[poss])
		comparo=ordos[poss]+2*vari
		ordenada=ordos[poss]-2*vari
		# print(comparo,ordenada)
	# print('kapasao')
	return [pendos[poss],ordos[poss]] 