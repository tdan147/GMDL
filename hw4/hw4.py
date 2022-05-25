import daft



pgm = daft.PGM([10,10], origin=[0,0], observed_style='shaded')

pgm.add_node(daft.Node(name='f1', content='f1',x=3, y=6.5, shape='rectangle', scale=1))
pgm.add_node(daft.Node(name='f2', content='f2',x=6, y=6.5, shape='rectangle', scale=1))
pgm.add_node(daft.Node(name='f3', content='f3',x=3, y=3.5, shape='rectangle', scale=1))

pgm.add_node(daft.Node(name='D', content='D',x=1, y=8, shape='ellipse', scale=1.75))
pgm.add_node(daft.Node(name='I', content='I',x=5, y=8, shape='ellipse', scale=1.75))
pgm.add_node(daft.Node(name='G', content='G',x=3, y=5, shape='ellipse', scale=1.75))
pgm.add_node(daft.Node(name='S', content='S',x=7, y=5, shape='ellipse', scale=1.75))
pgm.add_node(daft.Node(name='L', content='L',x=3, y=2, shape='ellipse', scale=1.75))

directed = False

pgm.add_edge('D', 'f1', directed)
pgm.add_edge('I', 'f1', directed)
pgm.add_edge('G', 'f1', directed)
pgm.add_edge('I', 'f2', directed)
pgm.add_edge('S', 'f2', directed)
pgm.add_edge('G', 'f3', directed)
pgm.add_edge('L', 'f3', directed)

pgm.render()
pgm.figure.savefig('figure.png', dpi=150)

