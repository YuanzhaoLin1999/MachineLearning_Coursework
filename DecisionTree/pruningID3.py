from Data_Class import T_data, DataSet, loadData
from pythonds import Stack
class Node:
    def __init__(self, dataset , attrset, basis = None, name = None, leaf=False, label=None):
        self.dataset = dataset
        self.attrset = attrset
        self.leaf = leaf
        self.basis = basis
        self.name = name
        self.label = label
        self.child = dict()
    def set_as_leaf(self,label):
        self.leaf = True
        self.label = label
    def recovery(self):
        self.leaf = False
        self.label = None
    def set_name(self, name):
        self.name = name
    def set_basis(self, basis):
        self.basis = basis    
    def set_child(self, node, feature):
        self.child[feature] = node

class TestDataError(ValueError):
    pass

#类：决策树分类器
class DesicionTreeClassifier:
    def __init__(self, dataset, attrset, critertion="entropy"):
        """criterion代表分类准则，有entropy和gini两个选项"""
        self.critertion = critertion
        self.root = Node(dataset, attrset, name='task')
    
    @staticmethod
    def best_label(node):
        dataset = node.dataset
        n = 0
        b_label = None
        for x in dataset.labels:
            if dataset.labels[x] > n:
                b_label = x
                n = dataset.labels[x]
        return b_label

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
                
    def generateTree(self, pruning='un', testData=None):
        """pruning 有 un pre post 三个选项，若需要剪枝，需有testData"""     
        def same_in_A(dataset,attrset):
            d = dataset.datas[0]
            for x in dataset.datas:
                for attr in attrset:
                    if d.data[attr] != x.data[attr]:
                        return False
            return True

        def gen_iter(node, pru, tD=testData):
            if_pruning = pru
            dataset = node.dataset
            attrset = node.attrset
            if if_pruning =='pre':#若需预剪枝，记录初始验证率
                node.set_as_leaf(label=self.best_label(node))
                ori_rate = self.clarrify_rate(tD)
                node.recovery()
            if len(dataset.labels) == 1: #样本全属一类
                label = list(dataset.labels.keys())[0]
                node.set_as_leaf(label)
                return
            elif not attrset or same_in_A(dataset, attrset):
                label = self.best_label(node)
                node.set_as_leaf(label)
                return
            attr, value, Dv = self.best_split(node)
            node.set_basis(attr)
            new_attrset = attrset.copy()
            new_attrset.remove(attr)
            if if_pruning == "un" or if_pruning == "post":
                for av in Dv:
                    ds = Dv[av]
                    av_node = Node(ds, new_attrset, name=av)
                    node.set_child(av_node, av)
                    gen_iter(av_node, pru=if_pruning,tD=testData)
            elif if_pruning == 'pre':
                for av in Dv:
                    ds = Dv[av]
                    av_node = Node(ds, new_attrset, name=av)
                    av_node.set_as_leaf(self.best_label(av_node))
                    node.set_child(av_node, av)
                new_rate = self.clarrify_rate(tD)
                if ori_rate > new_rate:
                    node.set_as_leaf(self.best_label(node))
                    node.child = dict()
                    return
                else:
                    for av in node.child:
                        av_node = node.child[av]
                        av_node.recovery()
                        gen_iter(av_node, pru = if_pruning, tD=testData)

        if (pruning == 'pre' or pruning == 'post') and testData is None:
            raise TestDataError
        gen_iter(self.root,pru=pruning,tD=testData)
        if pruning == 'post':
            ori_rate = self.clarrify_rate(testData)
            dfs = []           
            def postorder(node):
                if node.leaf:#叶节点不进入dfs序列
                    return
                nonlocal dfs
                for x in node.child:
                    ch_node = node.child[x]
                    postorder(ch_node)
                dfs.append(node)
            postorder(self.root)
            for post_node in dfs:
                post_node.set_as_leaf(self.best_label(post_node))
                new_rate = self.clarrify_rate(testData)
                if new_rate > ori_rate:
                    post_node.child = dict()
                    ori_rate = new_rate
                else:
                    post_node.recovery()
                    
        
    def clarrify(self, data):
        st = Stack()
        st.push(self.root)
        while st is not None:
            node = st.pop()
            if node.leaf:
                return data.label == node.label
            basis = node.basis
            feature = data.data[basis]
            st.push(node.child[feature])

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
    dataset, attrset = loadData("4.2_data.csv")
    testdataset, testattrset = loadData('4.2_test.csv')
    clf = DesicionTreeClassifier(dataset, attrset,critertion='gini')
    clf.generateTree(pruning='un')
    clf.export_tree('4.2_un.dot')
    clf2 = DesicionTreeClassifier(dataset, attrset)
    clf2.generateTree(pruning='pre',testData=testdataset)
    clf2.export_tree('4.2_pre.dot')
    clf3 = DesicionTreeClassifier(dataset, attrset)
    clf3.generateTree(pruning='post',testData= testdataset)
    clf3.export_tree('4.2_post.dot')