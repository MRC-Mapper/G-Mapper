"""
Modified visualization.py in Mapper Interactive.
"""
from graph import EnhancedGraph, Graph, AbstractGraph
from node import Sign, Node, EnhancedNode
try:
    from pyvis.network import Network
    from matplotlib import cm
    from matplotlib.colors import rgb2hex
    pyvis_available = True
except ImportError:
    pyvis_available = False
import matplotlib
import matplotlib.pyplot as plt
'''
Visualization functions.
'''

def node2compactrep(node, enhanced):
    interval_idx = node.interval_index
    cluster_idx = node.cluster_index
    s = f'Interval: {interval_idx} Cluster: {cluster_idx}'
    if enhanced:
        sign = node.sign.name.lower()
        s += f' Sign: {sign}'
    return s

def vis_graph(g, title:str, lens, save_loc, piechart_save=None, labels=None, cmap: str = 'jet', enhanced=False):
    """
    Very similar to pyvis_visualize. TODO: This definitely needs an overhaul.
    """

    def save_piechart(n):
        fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(aspect="equal"))
        c = cm.get_cmap('tab10', 10)
        colors = [c(i) for i in range(10)]
        curr = labels[n.members]
        counts = [curr[curr == i].shape[0] for i in range(10)]
        ax.pie(counts, colors=colors)
        plt.savefig(save_loc + piechart_save + n.short_string() + '.png', dpi=30)
        plt.close(fig)
        return piechart_save + n.short_string() + '.png'
    
    assert len(g.nodes) != 0
    nt = Network(notebook=True, height='1000px', width='500px', heading=title)
    nt.toggle_physics(True)
    color_map = cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=lens.min(), vmax=lens.max(), clip=True)
    color_mapping = cm.ScalarMappable(norm, color_map)
    get_c = lambda x: rgb2hex(color_mapping.to_rgba(x))
    for node in g.nodes:
        if labels is None:
            nt.add_node(node2compactrep(node, enhanced), label=' ', title=str(node), \
                color= get_c(lens[node.members].mean()))
        else:
            nt.add_node(node2compactrep(node, enhanced), label=' ', title=str(node), shape='image', image=save_piechart(node), size=60)
    for e in g.edges:
        n1, n2 = node2compactrep(e[0], enhanced), node2compactrep(e[1], enhanced)
        nt.add_edge(n1, n2)
    nt.repulsion(node_distance=250, central_gravity=0.5, spring_length=75, spring_strength=0.05, damping=0.09)
    nt.prep_notebook()
    nt.show(save_loc + title + '.html')


def pyvis_visualize(g: AbstractGraph, title:str, fname:str, enhanced:bool = False, notebook:bool = True, cmap: str = 'autumn', physics: bool=True):
    if not pyvis_available:
        raise ModuleNotFoundError('pyvis or matplotlib could not be found. This functionality is unavailable.')
    assert len(g.nodes) != 0
    nt = Network(notebook=notebook, height="500px", width="100%",heading=title)
    nt.toggle_physics(physics)
    color_map = cm.get_cmap(cmap)
    if enhanced:
        max_fn_val = max([g.function[n] for n in g.nodes])
        min_fn_val = min([g.function[n] for n in g.nodes])
        fn_range = max_fn_val - min_fn_val

    for node in g.nodes:
        nt.add_node(node2compactrep(node, enhanced), label=' ', title=str(node), \
            color= rgb2hex(color_map((g.function[node] - min_fn_val)/fn_range)[:3]) if enhanced else rgb2hex(cm.get_cmap('rainbow')(g.means[node])[:3])) # Modified - add rgb2hex(cm.get_cmap('rainbow')(g.means[node])[:3])). 
    
    for e in g.edges:
        n1, n2 = node2compactrep(e[0], enhanced), node2compactrep(e[1], enhanced)
        nt.add_edge(n1, n2)
    if notebook:
        nt.prep_notebook()
    nt.show(fname)
    # return nt


