

#%%
import itertools

from neuralogic.core import Template, Settings, R, V, Metadata, Activation, Aggregation
from neuralogic.core import Relation, Dataset, Data

dataset = Dataset()

story = [('donald', 'father', 'michael'), ('michael', 'sister', 'dorothy')]
target = ('donald', 'aunt', 'dorothy')


example = list(itertools.chain(
    (Relation.edge(u, r, v) for u, r, v in story),
    (Relation.rel(r) for r in ['aunt', 'father', 'sister']),
    (Relation.person(p) for p in ['donald', 'michael', 'dorothy'])
))

query = Relation.predict(*target)


template = Template()

template.add_rules([
    (R.person_embed(V.A)[3,] <= R.person(V.A)),
    # (R.rel_embed(V.B)[3,] <= R.rel(V.B))
])

#%%
from neuralogic.utils.visualize import draw_model
from neuralogic.utils.data import XOR_Vectorized
from neuralogic.core import Settings, Backend

model = template.build(Backend.JAVA, Settings())

draw_model(model) #, filename="../data/images/drawing.png")

#%%

template2 = Template()

template2.add_rules([
    *[(R.person_embed(p)[3,] <= R.person(p)) for p in ("dorothy", "donald", "michael")],
    # *[(R.rel_embed(r)[3,] <= R.rel(r)) for r in ("father", "sister", "aunt")]
])
model = template2.build(Backend.JAVA, Settings())

draw_model(model)

#%%

from neuralogic.utils.visualize import draw_model
from neuralogic.utils.data import XOR_Vectorized
from neuralogic.core import Settings, Backend


template, dataset = XOR_Vectorized()
model = template.build(Backend.JAVA, Settings())

# draw_model(model)
draw_model(model, filename="/Users/boris.rakovan/Desktop/school/thesis/code/data/images/drawing.pgn")
