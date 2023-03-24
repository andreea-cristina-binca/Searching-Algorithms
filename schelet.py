import random, math, time, sys
from _functools import reduce
from copy import copy, deepcopy
from builtins import isinstance
from resource import setrlimit, RLIMIT_AS, RLIMIT_DATA, RLIMIT_STACK, RLIM_INFINITY
from heapq import heappop, heappush

class NPuzzle:
	"""
	Reprezentarea unei stări a problemei și a istoriei mutărilor care au adus starea aici.
	
	Conține funcționalitate pentru
	- afișare
	- citirea unei stări dintr-o intrare pe o linie de text
	- obținerea sau ștergerea istoriei de mutări
	- obținerea variantei rezolvate a acestei probleme
	- verificarea dacă o listă de mutări fac ca această stare să devină rezolvată.
	"""

	NMOVES = 4
	UP, DOWN, LEFT, RIGHT = range(NMOVES)
	ACTIONS = [UP, DOWN, LEFT, RIGHT]
	names = "UP, DOWN, LEFT, RIGHT".split(", ")
	BLANK = ' '
	delta = dict(zip(ACTIONS, [(-1, 0), (1, 0), (0, -1), (0, 1)]))
	
	PAD = 2
	
	def __init__(self, puzzle : list[int | str], movesList : list[int] = []):
		"""
		Creează o stare nouă pe baza unei liste liniare de piese, care se copiază.
		
		Opțional, se poate copia și lista de mutări dată.
		"""
		self.N = len(puzzle)
		self.side = int(math.sqrt(self.N))
		self.r = copy(puzzle)
		self.moves = copy(movesList)
	
	def display(self, show = True) -> str:
		row = "-" * ((NPuzzle.PAD + 1) * self.side + 1)
		aslist = self.r
		
		slices = [aslist[ slice * self.side : (slice+1) * self.side ]  for slice in range(self.side)]
		s = ' |\n| '.join([' '.join([str(e).rjust(NPuzzle.PAD, ' ') for e in line]) for line in slices]) 
	
		s = ' ' + row + '\n| ' + s + ' |\n ' + row
		if show: print(s)
		return s
	def display_moves(self):
		print([names[a] if a is not None else None for a in moves])
		
	def print_line(self):
		return str(self.r)
	
	@staticmethod
	def read_from_line(line : str):
		list = line.strip('\n][').split(', ')
		numeric = [NPuzzle.BLANK if e == "' '" else int(e) for e in list]
		return NPuzzle(numeric)
	
	def clear_moves(self):
		"""Șterge istoria mutărilor pentru această stare."""
		self.moves.clear()
	
	def apply_move_inplace(self, move : int):
		"""Aplică o mutare, modificând această stare."""
		blankpos = self.r.index(NPuzzle.BLANK)
		y, x = blankpos // self.side, blankpos % self.side
		ny, nx = y + NPuzzle.delta[move][0], x + NPuzzle.delta[move][1]
		if ny < 0 or ny >= self.side or nx < 0 or nx >= self.side: return None
		newpos = ny * self.side + nx
		piece = self.r[newpos]
		self.r[blankpos] = piece
		self.r[newpos] = NPuzzle.BLANK
		self.moves.append(move)
		return self
	
	def apply_move(self, move : int):
		"""Construiește o nouă stare, rezultată în urma aplicării mutării date."""
		return self.clone().apply_move_inplace(move)

	def solved(self):
		"""Întoarce varianta rezolvată a unei probleme de aceeași dimensiune."""
		return NPuzzle(list(range(self.N))[1:] + [NPuzzle.BLANK])

	def verify_solved(self, moves : list[int]) -> bool:
		""""Verifică dacă aplicarea mutărilor date pe starea curentă duce la soluție"""
		return reduce(lambda s, m: s.apply_move_inplace(m), moves, self.clone()) == self.solved()

	def clone(self):
		return NPuzzle(self.r, self.moves)
	def __str__(self) -> str:
		return str(self.N-1) + "-puzzle:" + str(self.r)
	def __repr__(self) -> str: return str(self)
	def __eq__(self, other):
		return self.r == other.r
	def __lt__(self, other):
		return True
	def __hash__(self):
		return hash(tuple(self.r))

	def manhattan_distance(self, state1, state2):
		sum = 0
		for i in range(self.N):
			if state1[i] != ' ':
				sum += abs(i % self.side - state2.index(state1[i]) % self.side) \
						+ abs(i // self.side - state2.index(state1[i]) // self.side)
		return sum
	
	def linear_conflicts(self, state1, state2):
		counter = 0
		for row in range(0, self.N, self.side):
			for i in range(row, row + self.side - 1):
				if state1[i] != ' ':
					for j in range(i + 1, row + self.side):
						if state1[j] != ' ':
							if state1[i] > state1[j]:
								if state2.index(state1[i]) // self.side == state2.index(state1[j]) // self.side == row // self.side:
									counter += 1
									break
		for col in range(0, self.side):
			for i in range(col, col + self.side * (self.side - 1), self.side):
				if state1[i] != ' ':
					for j in range(i + self.side, col + self.side * (self.side), self.side):
						if state1[j] != ' ':
							if state1[i] > state1[j]:
								if state2.index(state1[i]) % self.side == state2.index(state1[j]) % self.side == col % self.side:
									counter += 1
									break
		return counter * 2 + self.manhattan_distance(state1, state2)

	def get_neighbours(self, puzzle):
		neighbours = []
		for act in self.ACTIONS:
			new_puzzle = puzzle.apply_move(act)
			if new_puzzle is not None:
				neighbours.append(new_puzzle)
		return neighbours

	def astar(self, start, end, h, limit):
		start_time = time.process_time()
		frontier = []
		heappush(frontier, (0 + h(start.r, end.r), start))
		discovered = {start: (None, 0)}
		while frontier and len(discovered) < limit:
			(total_cost, node) = heappop(frontier)
			if node == end:
				break
			(parent, cost_to_node) = discovered[node]
			for neigh in node.get_neighbours(node):
				neigh_cost = cost_to_node + 1
				if ((neigh not in discovered)) or (discovered[neigh][1] > neigh_cost):
					new_cost = neigh_cost + h(neigh.r, end.r)
					discovered[neigh] = (node, neigh_cost)
					heappush(frontier, (new_cost, neigh))
		end_time = time.process_time()
		if len(discovered) >= limit:
			return (-1, len(discovered), end_time - start_time)
		return (len(node.moves), len(discovered), end_time - start_time)

	def beam_search(self, start, end, h, B, limit):
		start_time = time.process_time()
		beam = {start: h(start.r, end.r)}
		discovered = {start: h(start.r, end.r)}
		while beam and len(discovered) < limit:
			successors = {}
			for s in beam.keys():
				succs = s.get_neighbours(s)
				for suc in succs:
					if suc not in discovered:
						successors[suc] = h(suc.r, end.r)
			for node in successors:
				if node == end:
					end_time = time.process_time()
					return (len(node.moves), len(discovered), end_time - start_time)
			sort_succs = sorted(successors, key=successors.get)
			selected = {}
			index = 0
			while index < B and index < len(sort_succs):
				selected[sort_succs[index]] = successors[sort_succs[index]]
				discovered[sort_succs[index]] = successors[sort_succs[index]]
				index += 1
			beam = copy(selected)
		end_time = time.process_time()
		return (-1, len(discovered), end_time - start_time)
	
	def gld_iteration(self, state, end, h, discrepancies, discovered, limit):
		successors = {}
		for s in self.get_neighbours(state):
			if s == end:
				return (True, state, len(discovered))
			if s not in discovered:
				successors[s] = h(s.r, end.r)
		if len(successors) == 0 or len(discovered) > limit:
			return (False, {}, -1)
		best = sorted(successors, key=successors.get)[0]
		best_value = successors[best]
		if discrepancies == 0:
			discovered[best] = best_value
			(value, path, moves) = self.gld_iteration(best, end, h, 0, discovered, limit)
			discovered.pop(best)
			return (value, path, moves)
		else:
			successors.pop(best)
			while len(successors):
				succ = sorted(successors, key=successors.get)[0]
				discovered[succ] = successors[succ]
				successors.pop(succ)
				(value, path, moves) = self.gld_iteration(succ, end, h, discrepancies - 1, discovered, limit)
				discovered.pop(succ)
				if value is True:
					return (value, path, moves)
		discovered[best] = best_value
		(value, path, moves) = self.gld_iteration(best, end, h, discrepancies, discovered, limit)
		discovered.pop(best)
		return (value, path, moves)

	def glds(self, start, end, h, limit):
		start_time = time.process_time()
		discrepancies = 0
		while True:
			discovered = {start: h(start.r, end.r)}
			(value, path, moves) = self.gld_iteration(start, end, h, discrepancies, discovered, limit)
			if value is True:
				end_time = time.process_time()
				return (len(path.moves), moves, end_time - start_time)
			discrepancies += 1

	def bld_iteration(self, level, end, h, B, discrepancies, discovered, limit):
		successors = {}
		for s in level:
			for suc in self.get_neighbours(s):
				if suc == end:
					return (True, suc, len(discovered))
				if suc not in discovered:
					successors[suc] = h(suc.r, end.r)
		if len(successors) == 0 or len(discovered) + min(B, len(successors)) > limit:
			return (False, {}, -1)
		sorted_succs = sorted(successors, key=successors.get)
		if discrepancies == 0:
			next_level = sorted_succs[:B]
			for lvl in next_level:
				discovered[lvl] = h(lvl.r, end.r)
			(value, state, moves) = self.bld_iteration(next_level, end, h, B, 0, discovered, limit)
			for lvl in next_level:
				discovered.pop(lvl)
			return (value, state, moves)
		else:
			explored = B
			while explored < len(successors):
				n = min(len(successors) - explored, B)
				next_level = sorted_succs[explored:explored + n]
				for lvl in next_level:
					discovered[lvl] = h(lvl.r, end.r)
				(value, state, moves) = self.bld_iteration(next_level, end, h, B, discrepancies - 1, discovered, limit)
				for lvl in next_level:
					discovered.pop(lvl)
				if value is True:
					return (value, state, moves)
				explored += len(next_level)
			next_level = sorted_succs[:B]
			for lvl in next_level:
				discovered[lvl] = h(lvl.r, end.r)
			(value, state, moves) = self.bld_iteration(next_level, end, h, B, discrepancies, discovered, limit)
			for lvl in next_level:
				discovered.pop(lvl)
			return (value, state, moves)
	
	def blds(self, start, end, h, B, limit):
		start_time = time.process_time()
		discovered = {start: h(start.r, end.r)}
		discrepancies = 0
		while True:
			(value, state, moves) = self.bld_iteration([start], end, h, B, discrepancies, discovered, limit)
			if value is True:
				end_time = time.process_time()
				return (len(state.moves), moves, end_time - start_time)
			discrepancies += 1


# generare
def genOne(side, difficulty):
	state = NPuzzle(list(range(side * side))[1:] + [NPuzzle.BLANK])
	for i in range(side ** difficulty + random.choice(range(side ** (difficulty//2)))):
		s = state.apply_move(random.choice(NPuzzle.ACTIONS))
		if s is not None: state = s
	state.clear_moves()
	return state

# problemele easy au fost generate cu dificultatile 4, 3, respectiv 2 (pentru marimile 4, 5, 6)
# celelalte probleme au toate dificultate 6

MLIMIT = 3 * 10 ** 9 # 2 GB RAM limit
setrlimit(RLIMIT_DATA, (MLIMIT, MLIMIT))
sys.setrecursionlimit(10000)
B = [1, 10, 50, 100, 500, 1000]
limits = [100000, 500000, 1000000]
# setrlimit(RLIMIT_DATA, (RLIM_INFINITY, RLIM_INFINITY))
# setrlimit(RLIMIT_STACK, (RLIM_INFINITY, RLIM_INFINITY))
# sys.setrecursionlimit(10**7)

f = open("files/problems4-easy.txt", "r")
input = f.readlines()
f.close()
problems4easy = [NPuzzle.read_from_line(line) for line in input]
f = open("files/problems5-easy.txt", "r")
input = f.readlines()
f.close()
problems5easy = [NPuzzle.read_from_line(line) for line in input]
f = open("files/problems6-easy.txt", "r")
input = f.readlines()
f.close()
problems6easy = [NPuzzle.read_from_line(line) for line in input]
f = open("files/problems4.txt", "r")
input = f.readlines()
f.close()
problems4 = [NPuzzle.read_from_line(line) for line in input]
f = open("files/problems5.txt", "r")
input = f.readlines()
f.close()
problems5 = [NPuzzle.read_from_line(line) for line in input]
f = open("files/problems6.txt", "r")
input = f.readlines()
f.close()
problems6 = [NPuzzle.read_from_line(line) for line in input]


print("* Astar Manhattan *")

print("4 easy:")
for i, p in enumerate(problems4easy):
	print(i + 1, p.astar(p, p.solved(), p.manhattan_distance, limits[0]))
print("5 easy:")
for i, p in enumerate(problems5easy):
	print(i + 1, p.astar(p, p.solved(), p.manhattan_distance, limits[1]))
print("6 easy:")
for i, p in enumerate(problems6easy):
	print(i + 1, p.astar(p, p.solved(), p.manhattan_distance, limits[2]))
print()


print("* Astar Linear Conflicts *")

print("4 easy:")
for i, p in enumerate(problems4easy):
	print(i + 1, p.astar(p, p.solved(), p.linear_conflicts, limits[0]))
print("5 easy:")
for i, p in enumerate(problems5easy):
	print(i + 1, p.astar(p, p.solved(), p.linear_conflicts, limits[1]))
print("6 easy:")
for i, p in enumerate(problems6easy):
	print(i + 1, p.astar(p, p.solved(), p.linear_conflicts, limits[2]))
print()
print()

print("* Beam Search Manhattan *")

for b in B:
	print("B =", b)
	print("4 easy:")
	for i, p in enumerate(problems4easy):
		print(i + 1, p.beam_search(p, p.solved(), p.manhattan_distance, b, limits[0]))
	print("5 easy:")
	for i, p in enumerate(problems5easy):
		print(i + 1, p.beam_search(p, p.solved(), p.manhattan_distance, b, limits[1]))
	print("6 easy:")
	for i, p in enumerate(problems6easy):
		print(i + 1, p.beam_search(p, p.solved(), p.manhattan_distance, b, limits[2]))
	print()
print()

for b in B:
	print("B =", b)
	print("4 hard:")
	for i, p in enumerate(problems4):
		print(i + 1, p.beam_search(p, p.solved(), p.manhattan_distance, b, limits[0]))
	print("5 hard:")
	for i, p in enumerate(problems5):
		print(i + 1, p.beam_search(p, p.solved(), p.manhattan_distance, b, limits[1]))
	print("6 hard:")
	for i, p in enumerate(problems6):
		print(i + 1, p.beam_search(p, p.solved(), p.manhattan_distance, b, limits[2]))
	print()
print()


print("* Beam Search Linear Conflicts *")

for b in B:
	print("B =", b)
	print("4 easy:")
	for i, p in enumerate(problems4easy):
		print(i + 1, p.beam_search(p, p.solved(), p.linear_conflicts, b, limits[0]))
	print("5 easy:")
	for i, p in enumerate(problems5easy):
		print(i + 1, p.beam_search(p, p.solved(), p.linear_conflicts, b, limits[1]))
	print("6 easy:")
	for i, p in enumerate(problems6easy):
		print(i + 1, p.beam_search(p, p.solved(), p.linear_conflicts, b, limits[2]))
	print()
print()

for b in B:
	print("B =", b)
	print("4 hard:")
	for i, p in enumerate(problems4):
		print(i + 1, p.beam_search(p, p.solved(), p.linear_conflicts, b, limits[0]))
	print("5 hard:")
	for i, p in enumerate(problems5):
		print(i + 1, p.beam_search(p, p.solved(), p.linear_conflicts, b, limits[1]))
	print("6 hard:")
	for i, p in enumerate(problems6):
		print(i + 1, p.beam_search(p, p.solved(), p.linear_conflicts, b, limits[2]))
	print()
print()


print("* GLDS Manhattan *")

print("4 easy:")
for i, p in enumerate(problems4easy):
	print(i + 1, p.glds(p, p.solved(), p.manhattan_distance, limits[0]))
print("5 easy:")
for i, p in enumerate(problems5easy):
	print(i + 1, p.glds(p, p.solved(), p.manhattan_distance, limits[1]))
print("6 easy:")
for i, p in enumerate(problems6easy):
	print(i + 1, p.glds(p, p.solved(), p.manhattan_distance, limits[2]))
print()

print("4 hard:")
for i, p in enumerate(problems4):
	print(i + 1, p.glds(p, p.solved(), p.manhattan_distance, limits[0]))
print("5 hard:")
for i, p in enumerate(problems5):
	print(i + 1, p.glds(p, p.solved(), p.manhattan_distance, limits[1]))
print("6 hard:")
for i, p in enumerate(problems6):
	print(i + 1, p.glds(p, p.solved(), p.manhattan_distance, limits[2]))
print()


print("* GLDS Linear Conflicts *")

print("4 easy:")
for i, p in enumerate(problems4easy):
	print(i + 1, p.glds(p, p.solved(), p.linear_conflicts, limits[0]))
print("5 easy:")
for i, p in enumerate(problems5easy):
	print(i + 1, p.glds(p, p.solved(), p.linear_conflicts, limits[1]))
print("6 easy:")
for i, p in enumerate(problems6easy):
	print(i + 1, p.glds(p, p.solved(), p.linear_conflicts, limits[2]))
print()

print("4 hard:")
for i, p in enumerate(problems4):
	print(i + 1, p.glds(p, p.solved(), p.linear_conflicts, limits[0]))
print("5 hard:")
for i, p in enumerate(problems5):
	print(i + 1, p.glds(p, p.solved(), p.linear_conflicts, limits[1]))
print("6 hard:")
for i, p in enumerate(problems6):
	print(i + 1, p.glds(p, p.solved(), p.linear_conflicts, limits[2]))
print()


print("* BLDS Manhattan *")

for b in B:
	print("B =", b)
	print("4 easy:")
	for i, p in enumerate(problems4easy):
		print(i + 1, p.blds(p, p.solved(), p.manhattan_distance, b, limits[0]))
	print("5 easy:")
	for i, p in enumerate(problems5easy):
		print(i + 1, p.blds(p, p.solved(), p.manhattan_distance, b, limits[1]))
	print("6 easy:")
	for i, p in enumerate(problems6easy):
		print(i + 1, p.blds(p, p.solved(), p.manhattan_distance, b, limits[2]))
	print()
print()

for b in B:
	if b == 1:
		continue
	print("B =", b)
	print("4 hard:")
	for i, p in enumerate(problems4):
		print(i + 1, p.blds(p, p.solved(), p.manhattan_distance, b, limits[0]))
	print("5 hard:")
	for i, p in enumerate(problems5):
		print(i + 1, p.blds(p, p.solved(), p.manhattan_distance, b, limits[1]))
	print("6 hard:")
	for i, p in enumerate(problems6):
		print(i + 1, p.blds(p, p.solved(), p.manhattan_distance, b, limits[2]))
	print()
print()


print("* BLDS Linear Conflicts *")

for b in B:
	print("B =", b)
	print("4 easy:")
	for i, p in enumerate(problems4easy):
		print(i + 1, p.blds(p, p.solved(), p.linear_conflicts, b, limits[0]))
	print("5 easy:")
	for i, p in enumerate(problems5easy):
		print(i + 1, p.blds(p, p.solved(), p.linear_conflicts, b, limits[1]))
	print("6 easy:")
	for i, p in enumerate(problems6easy):
		print(i + 1, p.blds(p, p.solved(), p.linear_conflicts, b, limits[2]))
	print()
print()

for b in B:
	print("B =", b)
	print("4 hard:")
	for i, p in enumerate(problems4):
		print(i + 1, p.blds(p, p.solved(), p.linear_conflicts, b, limits[0]))
	print("5 hard:")
	for i, p in enumerate(problems5):
		print(i + 1, p.blds(p, p.solved(), p.linear_conflicts, b, limits[1]))
	print("6 hard:")
	for i, p in enumerate(problems6):
		print(i + 1, p.blds(p, p.solved(), p.linear_conflicts, b, limits[2]))
	print()
print()
