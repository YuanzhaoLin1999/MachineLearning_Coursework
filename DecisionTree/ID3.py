from Data_Class import T_data, DataSet, loadData
from pythonds import Stack
class Node:
    def __init__(self, dataset , attrset, name = None, leaf=False, label=None):
        self.dataset = dataset
        self.attrset = attrset
        self.leaf = leaf
        self.name = name
        self.label = label
        self.child = dict()
    def set_as_leaf(self,label):
        self.leaf = True
        self.label = label
    def set_name(self, name):
        self.name = name    
    def set_child(self, node, feature):
        self.child[feature] = node
#类：决策树分类器
class DesicionTreeClassifier:
    def __init__(self, dataset, attrset, critertion="entropy"):
        """criterion代表分类准则，有entropy和gini两个选项"""
        self.critertion = critertion
        self.root = Node(dataset, attrset, name='task')
    
    @staticmethod
    def is_numeric(data):
        return isinstance(data, int) or isinstance(data, float)

    def best_split(self, node):
        dataset = node.dataset
        attrset = node.attrset
        if self.critertion == "entropy":
            entropy = 0
            av, division = None, None
            for attr in attrset:
                infor_gain, Dv = dataset.info_gain(attr)
                if infor_gain > entropy:
                    entropy = infor_gain
                    av = attr
                    division = Dv
            return av, entropy, division
        elif self.critertion == "gini":
            gini_index = 1
            av, division = None, None
            for attr in attrset:
                gini, Dv = dataset.gini_index(attr)
                if gini < gini_index:
                    gini_index = gini
                    av = attr
                    division = Dv
            return av, gini_index, division
                
    def generateTree(self, pruning='un'):     
        def same_in_A(dataset,attrset):
            d = dataset.datas[0]
            for x in dataset.datas:
                for attr in attrset:
                    if d.data[attr] != x.data[attr]:
                        return False
            return True
        def gen_iter(node, pru):
            if_pruning = pru
            dataset = node.dataset
            attrset = node.attrset
            if len(dataset.labels) == 1: #样本全属一类
                label = list(dataset.labels.keys())[0]
                node.set_as_leaf(label)
                return
            elif not attrset or same_in_A(dataset, attrset):
                n = 0
                label = None
                for x in dataset.labels:
                    if dataset.labels[x] > n:
                        label = x
                node.set_as_leaf(label)
                return
            attr, value, Dv = self.best_split(node)
            new_attrset = attrset.copy()
            new_attrset.remove(attr)
            for av in Dv:
                ds = Dv[av]
                av_node = Node(ds, new_attrset, name=av)
                node.set_child(av_node, av)
                gen_iter(av_node, pru=if_pruning)
        gen_iter(self.root,pru=pruning)
    
    def clarrify(self, data):
        def iter_clarrify(data, node):
            if node.leaf:
                return node.label == data.label
            basis = node.basis
            feature = data.data[basis]
            iter_clarrify(data, node.child[feature])

        iter_clarrify(data, self.root)

    def clarrify_rate(self, dataset):
        n = 0
        N = len(dataset.datas)
        for data in dataset.datas:
            if self.clarrify(data):
                n += 1
        return n/N
    def export_tree(self, filename):
        f = open(filename, 'w')
        f.write('digraph Tree { \n node [shape=box] ;\n')
        st = Stack()
        st.push(self.root)
        while not st.isEmpty():
            node = st.pop()
            name, label = node.name, node.label
            if node.leaf:
                content = '[label="{}",name="{}"];\n'.format(label, name)
                f.write(name+content)
                continue
            else:
                content = '[name="{}",basis="{}"];\n'.format(name,node.basis)
                f.write(name+content)
                for x in node.child:
                    child_node = node.child[x]
                    st.push(child_node)#保证栈内总是Node类型
                    f.write('{} -> {} ;\n'.format(name, child_node.name))
                continue
        f.write('}')
        f.close()


if __name__ == "__main__":
    import graphviz
    dataset, attrset = loadData("4.2.csv")
    clf = DesicionTreeClassifier(dataset, attrset)
    clf.generateTree()
    clf.export_tree('4.2_en.dot')
    clf2 = DesicionTreeClassifier(dataset, attrset,critertion='gini')
    clf2.generateTree()
    clf2.export_tree('4.2_gini.dot')
