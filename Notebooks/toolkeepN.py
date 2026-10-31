# This code extracts the top N basis sounds.

projtopN = (palate[:,wind])[:,:keeptopN.value] @ (w[wind])[:keeptopN.value]

print("N = {}".format(keeptopN.value))
print('Instrument/Note', '\t', 'w')

for i,a in zip( np.array(bankbasis)[wind][:keeptopN.value], w[wind][:keeptopN.value]):
    print(i, '\t', a)
print()
print('w sum:', w.sum())
print('w_kept sum:', w[wind][:keeptopN.value].sum(), ' w_kept fraction:', w[wind][:keeptopN.value].sum()/w.sum() )
print('w_not_kept sum:', w[wind][keeptopN.value:].sum(), ' w_not_kept fraction:', w[wind][keeptopN.value:].sum()/w.sum() )

(pltw, plth) = plt.rcParams['figure.figsize']
plt.rcParams['figure.figsize'] = (8,3)

wmax = w[wind][0]
if wmax > 1:
    ymax = wmax+0.1
else:
    ymax = 1
plt.plot(range(keeptopN.value), w[wind][:keeptopN.value], 'o')
plt.ylim(0,ymax)
plt.xticks(range(keeptopN.value), np.array(bankbasis)[wind][:keeptopN.value], rotation=60)
plt.grid()
plt.tight_layout()
plt.show()

###

plt.rcParams['figure.figsize'] = (17,4)

plt.plot(range(len(target)), target, label='target')
plt.plot(range(len(target)), projtopN, label='top N={}'.format(keeptopN.value), alpha=0.7)

plt.legend()
plt.grid()
plt.yscale('log')
#plt.xscale('log')
plt.xlim(20,len(target))        
plt.show()        