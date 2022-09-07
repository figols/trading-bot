from contra import ALPHA
import pandas as pd
from collections import Counter
import requests, json, random, math, time, pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import funcions as fcs

def pickleLoader(pklFile):
    try:
        while True:
            yield pickle.load(pklFile)
    except EOFError:
        pass

start_time = time.time()
#problema en 3671
data=fcs.download('AAPL','daily',vela=True)
parells=[[480,980],[668,1168],[1409, 1909],[1749, 2249],[2185, 2885],[2744, 3244],[3170,3670],[3675,4175],[4085,4585],[4790, 5240]]
parells=[[3675,4175]]
# parells=[[480,980],[668,1168],[1409, 1909],[1749, 2249],[2185, 2885],[2744, 3244],[3170,3670],[3675,4175],[4085,4585],[4790, 5240]]
for par in parells:
	tros=data.copy()
	pd.set_option('display.max_columns', None)
	# a,b=[3672, 5000]
	# print([a,b])
	# print([tros.iloc[a]['Date'],tros.iloc[b]['Date']])
	print([par[0],par[1]])
	[a,b]=[par[0],par[1]]
	dates=[tros.iloc[a]['Date'],tros.iloc[b]['Date']]
	print(dates)
	tros=tros[a:b] #obtindre tros sense error
	tros=tros.reset_index(drop=True)
	tros['Mig']=(tros['High']+tros['Low'])/2

	'indicadors'
	# tros['TR']=fcs.TR(tros)
	# tros['38']=fcs.trenta(tros)
	# tros['Golf']=fcs.engulf(tros)
	# tros['tanca']=fcs.cierra(tros)
	# tros['ATR']=fcs.ATR(tros)
	# tros['ATR %']=round(tros['ATR']/tros['Mig']*100,2)
	# tros['RSI']=fcs.RSI(tros)
	# n=20
	# tros[f'MA {n}']=fcs.MA(tros,n)

	'mostrar algunes columnes de tros'
	# colons=list(tros.columns.values.tolist())
	# print(colons[0:5]+colons[6:])
	# for j in range(0,150,50):
	# 	print(tros[colons[1:5]+colons[8:10]+colons[12:15]][j:j+50])

	colors=['ro','go','bo','yo','mo','co','ko','rs','gs','bs','ys','ms','cs','ks',
	'r*','g*','b*','y*','m*','c*','k*']
	colores=['r','g','b','y','m','c','k']

	[nou,contador,tot,nombre,marges]=fcs.tamany(tros)
	# print('nou',nou)
	# print('nombre',nombre)
	# print('marges',marges)
	# for term in tot:
	# 	print('term',term)


	[Giga,Mega,Ea]=fcs.suport(tros,nou,'Max')
	# print('Giga',Giga)
	# print('Mega',Mega)
	# print('Ea',Ea)
	# print('Cond',Cond)

	'GRÀFIC suport/resistència'
	# for j in range(len(Ea)):
	# 	plt.figure(j) #mostrar l'estructura de tros['Mig']
	# 	wm = plt.get_current_fig_manager()
	# 	wm.window.state('zoomed')
	# 	plt.plot(tros['Date'],tros['Mig'],'k--',label=[a,b])
	# 	plt.plot(tros['Date'],tros['High'])
	# 	plt.plot(tros['Date'],tros['Low'])
	# 	plt.xticks(np.arange(0, len(tros['Date'])+1,10))
	# 	plt.xticks(rotation=45)
	# 	plt.xticks(fontsize=11)
	# 	plt.legend() # (xpos,ypos,xlen,ylen)
	# 	k=0
	# 	for i in range(len(Ea[j])):
	# 		if k==len(colores):
	# 			k=0
	# 		rect = patches.Rectangle((Giga[j][i][0],Ea[j][i]), Giga[j][i][-1]-Giga[j][i][0], 3, linewidth=1, edgecolor=colores[k], facecolor='none')
	# 		plt.gca().add_patch(rect)
	# 		k+=1
	# 	plt.show(block=False)
	# 	plt.pause(0.0001)
	# plt.show()
		

	[tot_min_quads,sak,trosses,macho,mache,borra]=fcs.tends_complet(nou,marges,0,purga=True)
	# print('tot_min_quads',tot_min_quads)
	# for term in trosses:
	# 	print('trosses',term)
	# print('sak',sak)
	prec,tol=0.95,1
	# print('prec',prec)

	estr=[fcs.estructura(tros,trosses,num,prec) for num in range(len(trosses))]
	# print('estr',estr)
	estre=[estr[i] for i in mache]
	planet=[term for termo in estre for term in termo]
	# print('planet',planet)

	[planet,pain]=fcs.neteja(planet,tros)
	print('pain',pain)
	# print('planet',planet)

	# with open('dades.pkl','ab') as file:
	# 	lista=[[a,b],dates,pain]
	# 	pickle.dump(lista, file)

	#pendent, ord. origen, angle, r2, asc o desc


	print("--- %s seconds ---" % (time.time() - start_time))

	'GRÀFIC'

	plt.figure() #mostrar l'estructura de tros['Mig']
	wm = plt.get_current_fig_manager()
	wm.window.state('zoomed')
	plt.plot(tros['Date'],tros['Mig'],'k--',label=[prec,tol,a,b])
	# plt.plot(tros['Date'],tros['High'])
	# plt.plot(tros['Date'],tros['Low'])
	for j in range(len(planet)):
		oa=np.array(range(planet[j][5],planet[j][6]+1))
		# print('eeaa',[planet[j][5],planet[j][6]+1])
		# print(oa)
		# print('puto',planet[j][0]*oa+np.ones((abs(planet[j][5]-planet[j][6])+1,), dtype=int)*planet[j][1])
		plt.plot(oa,planet[j][0]*oa+np.ones((abs(planet[j][5]-planet[j][6])+1,), dtype=int)*planet[j][1],linewidth=3)
	k=0
	for term in pain:
		if k==len(colors):
	 			k=0
		plt.plot(tros.iloc[term[0][3]]['Date'],term[0][1],colors[k])
		plt.plot(tros.iloc[term[-1][4]]['Date'],term[-1][2],colors[k])
		k+=1
	plt.xticks(np.arange(0, len(tros['Date'])+1,10))
	plt.xticks(rotation=45)
	plt.xticks(fontsize=11)
	plt.legend()
	plt.show(block=False)
	plt.pause(0.0001)
	plt.show()