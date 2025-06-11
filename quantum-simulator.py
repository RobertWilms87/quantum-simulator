import curses
import sys
import textwrap
import numpy as np
import matplotlib.pyplot as plt

def main(stdscr, n):
	curses.curs_set(1)
	curses.set_escdelay(1)
	stdscr.keypad(True)
	n = max(1, min(n, 36))
	initial_array=np.zeros(n)
	legend1 = 'H: Hadamard-gate, X: X-gate, Y: Y-gate, Z: Z-gate, T: T-gate, P: Phase gate (number indicates phase in °), C: Control bit (number indicates target bit),  M: Measurement,'
	legend2 = 'ESC: Quit, SPACE: Run, RETURN: Histogram, DEL: Delete'
	rows, cols = stdscr.getmaxyx()
	legend_lines = textwrap.wrap(legend1, width=cols) + textwrap.wrap(legend2, width=cols)
	legend_height = len(legend_lines) + 1
	if legend_height < 1:
		legend_height = 1
	usable_rows = rows - legend_height
	if usable_rows < 1:
		usable_rows = 1
	num_width = 7
	spacer = 1
	usable_width = cols - (num_width + spacer)-7
	if usable_width < 1:
		usable_width = 1
	labels = []
	lab=np.full(n, "", dtype=str)
	for i in range(n-1,-1,-1):
		if i <= 9:
			labels.append('q_'+str(i)+' |0>')
			lab[n-i-1]=str(i)
		else:
			labels.append('q_'+chr(ord('A') + (i - 10))+' |0>')
			lab[n-i]=chr(ord('A') + (i - 10))
	lines = [['|----' for _ in range(int(usable_width/5))] for _ in range(n)]
	for idx, text in enumerate(legend_lines):
		if idx >= rows:
			break
		stdscr.addstr(idx, 0, text)
	for i, line in enumerate(lines):
		if i >= usable_rows:
			break
		label = labels[i]
		row = legend_height + i
		stdscr.addstr(row, 0, label + ' ' + ''.join(line)+'|-M-- 0')
	lines = [['-' for _ in range(int(usable_width/5))] for _ in range(n)]
	y, x = 0, 1
	stdscr.move(legend_height + y, num_width + spacer + x)
	while True:
		key = stdscr.getch()
		if key==27:
			break
		if key in (10, 13, 32, curses.KEY_ENTER):
			output=output_state(n,initial_array,lines)
			instance=np.random.choice(a=2**n,p=probs(output))
			for i in range(n):
				stdscr.addch(legend_height + i, num_width + spacer + usable_width-usable_width%5+6, format(instance, f"0{n}b")[i])
			if key in (10, 13, curses.KEY_ENTER):
				plt.bar(np.arange(2**n),probs(output))
				plt.xticks(range(2**n))
				plt.xlim(0, 2**n)
				plt.ylim(0,1)
				plt.ylabel('Probability')
				plt.xlabel('Basis state')
				plt.show()
		elif key == curses.KEY_LEFT and x > 1:
			if x % 5 == 1:
				x -= 2
			else:
				x -= 1
		elif key == curses.KEY_LEFT and x==1:
			x=-3
		elif key == curses.KEY_RIGHT and x>0 and x < int(usable_width/5)*5 - 1:
			if x % 5 == 4:
				x += 2
			else:
				x += 1
		elif key== curses.KEY_RIGHT and x==-3:
			x=1
		elif key == curses.KEY_UP and y > 0:
			y -= 1
		elif key == curses.KEY_DOWN and y < n - 1 and y < usable_rows - 1:
			y += 1
		elif key == 9 and x < int(usable_width/5)*5 -5 :
			x += 5
		if x==-3 and key in (ord('0'), ord('1')):
			ch=chr(key)
			initial_array[y]=int(ch)
			stdscr.addch(legend_height + y, num_width + spacer + x, ch)	
		if x % 5 in range(1,5) and x>0:
			if key in (127, curses.KEY_DC, ord('-')):
						lines[y][int(x/5)]='-'
						stdscr.addch(legend_height + y, num_width + spacer + x-x%5+1, '-')
						stdscr.addch(legend_height + y, num_width + spacer + x-x%5+2, '-')	
						stdscr.addch(legend_height + y, num_width + spacer + x-x%5+3, '-')	
						stdscr.addch(legend_height + y, num_width + spacer + x-x%5+4, '-')
			if lines[y][int(x/5)][0]=='P' and x%5!=1 and key in (list(range(48, 58))):
				ch = chr(key)
				lines[y][int(x/5)]=lines[y][int(x/5)][:(x%5-1)]+ch+lines[y][int(x/5)][(x%5):]
				stdscr.addch(legend_height + y, num_width + spacer + x, ch)				
			elif lines[y][int(x/5)][0]=='C' and x%5!=2 and (key in (list(range(48, 58)) + list(range(97, 123)))[:n] or key in (list(range(48, 58)) + list(range(65, 91)))[:n]):
				ch = chr(key).upper()
				if ch!=lab[y]:
					lines[y][int(x/5)]='C'+ch
					stdscr.addch(legend_height + y, num_width + spacer + x-x%5+3, ch)
			else:
				if key in (ord('H'), ord('h'), ord('X'), ord('x'), ord('Y'), ord('y'), ord('Z'), ord('z'), ord('T'), ord('t')):
					ch = chr(key).upper()
					lines[y][int(x/5)]=ch
					stdscr.addch(legend_height + y, num_width + spacer + x-x%5+1, '-')	
					stdscr.addch(legend_height + y, num_width + spacer + x-x%5+2, ch)	
					stdscr.addch(legend_height + y, num_width + spacer + x-x%5+3, '-')	
					stdscr.addch(legend_height + y, num_width + spacer + x-x%5+4, '-')	
				elif key in (ord('C'), ord('c')):
					lines[y][int(x/5)]='C-'
					stdscr.addch(legend_height + y, num_width + spacer + x-x%5+1, '-')	
					stdscr.addch(legend_height + y, num_width + spacer + x-x%5+2, 'C')	
					stdscr.addch(legend_height + y, num_width + spacer + x-x%5+3, '-')	
					stdscr.addch(legend_height + y, num_width + spacer + x-x%5+4, '-')
				elif key in (ord('P'), ord('p')):
					lines[y][int(x/5)]='P000'
					stdscr.addch(legend_height + y, num_width + spacer + x-x%5+1, 'P')							
					stdscr.addch(legend_height + y, num_width + spacer + x-x%5+2, '0')	
					stdscr.addch(legend_height + y, num_width + spacer + x-x%5+3, '0')	
					stdscr.addch(legend_height + y, num_width + spacer + x-x%5+4, '0')
		stdscr.move(legend_height + y, num_width + spacer + x)
		stdscr.refresh()

def output_state(NQ, initial_array,circuit):
	X=np.array([[0,1],[1,0]])
	Y=np.array([[0,-1j],[1j,0]])
	Z=np.array([[1,0],[0,-1]])
	T=phase(np.pi/4)
	I=np.array([[1,0],[0,1]])
	H=1/np.sqrt(2)*np.array([[1,1],[1,-1]])
	depth=len(circuit[0])
	s=''.join(str(int(b)) for b in initial_array)
	initial_state = basis_state(int(s, 2),NQ)
	Q=np.array([[1,0],[0,0]])
	R=np.array([[0,0],[0,1]])
	extended_circuit = [[[circuit[j][i] for j in range(n)]] for i in range(depth)]
	for i in range(depth):
		for j in range(n):
			for l in range(len(extended_circuit[i])):
				if extended_circuit[i][l][j][0]=='C':
					if extended_circuit[i][l][j][1]!='-':
						newcircuit=[extended_circuit[i][l][j] for j in range(n)]
						newcircuit[j]='Q'
						k=int(extended_circuit[i][l][j][1],n)
						newcircuit[n-1-k]='-'
						extended_circuit[i].append(newcircuit)
						extended_circuit[i][l][j]='R'
	U_total=np.identity(2**NQ)
	for i in range(depth):
		Usum=0
		for l in range(len(extended_circuit[i])):
			U=1		
			for j in range(n):
				if extended_circuit[i][l][j]=='X':
					U=np.kron(U,X)
				if extended_circuit[i][l][j]=='Y':
					U=np.kron(U,Y)
				if extended_circuit[i][l][j]=='Z':
					U=np.kron(U,Z)
				if extended_circuit[i][l][j]=='T':
					U=np.kron(U,T)
				if extended_circuit[i][l][j]=='-' or extended_circuit[i][l][j][0]=='C':
					U=np.kron(U,I)
				if extended_circuit[i][l][j]=='H':
					U=np.kron(U,H)
				if extended_circuit[i][l][j]=='Q':
					U=np.kron(U,Q)
				if extended_circuit[i][l][j]=='R':
					U=np.kron(U,R)
				if extended_circuit[i][l][j][0]=='P':
					phi=2*np.pi*int(extended_circuit[i][l][j][1:])/360
					U=np.kron(U,phase(phi))
			Usum=Usum+U
		U_total=Usum@U_total
	return U_total@initial_state
	
def phase(phi):
	return np.array([[1,0],[0,np.exp(1j*phi)]])
	
def probs(state):
    nstate=normed_state(state)
    return (nstate.conjugate()*nstate).real

def basis_state(i,NQ):
    return np.identity(2**NQ)[i]
    
def normed_state(state):
    if np.linalg.norm(state)!=0:
        return state/np.linalg.norm(state)
    else:
        return "State is not normalizable, it is zero"

if __name__ == '__main__':
	try:
		raw = input('Number of qubits: ').strip()
		if raw == '':
			n = 4
		else:
			n = int(raw)
	except ValueError:
		print('Ungültige Eingabe.')
		sys.exit(1)

	curses.wrapper(lambda stdscr: main(stdscr, n))

