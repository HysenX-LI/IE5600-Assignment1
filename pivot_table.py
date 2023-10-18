import sys


class data_tree_node():
    def __init__(self, name = None, val = None, data_index = set(), depth = 0):
        self.name = name
        self.val = val
        self.data_index = data_index
        self.depth = depth
        self.son = []

        self.aggregate_position = 0

    def add_son(self,node):
        self.son.append(node)

class pivot_table():
    def __init__(self, data, columns : list) -> None:
        self.data = data
        self.columns = columns
        self.col_names = []
        self.row_names = []

        self._init_tree()

        self.aggregate_method = []
        self.aggregate_name = []
        self.aggregate_method_num = 0
        self.aggregate_position = 0

    # 初始化树
    def _init_tree(self):
        self.col_root = data_tree_node(data_index=set([i for i in range(len(data))]),val = "总计")
        self.row_root = data_tree_node(data_index=set([i for i in range(len(data))]),val = "总计")
        self.row_leaves = self._get_leaves(self.row_root)
        self.col_leaves = self._get_leaves(self.col_root)

    # 将树初始化后，根据行列字段更新树
    def _updata_tree(self):
        self._init_tree()
        for name in self.col_names:
            self._add_col(name)
        for name in self.row_names:
            self._add_row(name)     
        self._get_print_table_header()
        self._get_print_table_val()
    
    # 新增一个列字段
    def add_col(self, name):
        self.col_names.append(name)
        self._add_col(name)
    
    # 新增一个行字段
    def add_row(self, name):
        self.row_names.append(name)
        self._add_row(name)
    
    # 删除一个列字段，懒得写删写节点了，所以直接把字段删了然后重新建一遍树
    def del_col(self, name):
        self.col_names.remove(name)
        self._init_tree()
        self._updata_tree()
    
    # 删除一个行字段
    def del_row(self, name):
        self.row_names.remove(name)
        self._init_tree()
        self._updata_tree()

    
    def add_agg(self, name, method):
        """
        增加一个值
        name: str 所要增加的值的字段名
        method: str 所要增加的操作名，可选操作有:
            sum, ave, max, min, count, std
        你也可以自己增加操作了，写一个_aggregate_xxx的函数
        然后在_get_aggregate_method函数里把字符串匹配一下
        """
        self.aggregate_method.append(method)
        self.aggregate_name.append(name)
        self.aggregate_method_num += 1
        self._get_print_table_header()
        self._get_print_table_val()
    
    # 删除一个值，参数同上
    def del_agg(self, name, method):
        for i in range(self.aggregate_method_num):
            if self.aggregate_method[i] == method and self.aggregate_name[i] == name:
                self.aggregate_method.pop(i)
                self.aggregate_name.pop(i)
                self.aggregate_method_num -= 1
                break
        self._get_print_table_header()
        self._get_print_table_val()

    # 设置值作用在列上还是行上，列上值为0， 行上值为1， 默认为0
    def set_agg_position(self, pos):
        self.aggregate_position = 1
    
    
    # 使用dfs获得树的叶子节点的集合
    def _get_leaves(self, root_node : data_tree_node):
        leaves = []
        def dfs(node):
            if len(node.son) == 0: # 是叶子节点
                leaves.append(node)
            else:
                for son in node.son:
                    self.dfs(son)
        dfs(root_node)
        return leaves
    
    # 依据增加的字段名称，将树向下划分一层，并返回新的叶子节点
    def _add_val(self, leaves, add_name):
        new_leaves = []
        add_name_ind = self.columns.index(add_name)
        for node in leaves:
            category = set([self.data[i][add_name_ind] for i in node.data_index])
            category = list(category)
            category.sort()

            for val in category:
                temp_index = []
                for i in node.data_index:
                    if self.data[i][add_name_ind] == val:
                        temp_index.append(i)
                new_node = data_tree_node(add_name, val, set(temp_index), node.depth+1)
                node.add_son(new_node) 
                new_leaves.append(new_node)
        return new_leaves
    
    # 内置的增加一个列字段函数
    def _add_col(self, col_name):
        new_leaves = self._add_val(self.col_leaves, col_name)
        self.col_leaves = new_leaves
    
    # 内置的增加一个行字段函数
    def _add_row(self, row_name):
        new_leaves = self._add_val(self.row_leaves, row_name)
        self.row_leaves = new_leaves

    # 计算最大值
    def _aggregate_max(self,data):
        return max(data)
    
    # 计算最小值
    def _aggregate_min(self,data):
        return min(data)
    
    # 计数
    def _aggregate_count(self,data):
        return len(set(data))
    
    # 计算求和
    def _aggregate_sum(self,data):
        return sum(data)

    # 计算平均值
    def _aggregate_ave(self,data):
        return sum(data)/len(data)

    # 计算标准差
    def _aggregate_std(self,data):
        ave = sum(data)/len(data)
        return sum([(val-ave)**2 for val in data])**0.5

    # 把聚合操作的字符串映射成对应的方法
    def _get_aggregate_method(self, method_str):
        if method_str == "sum":
            return self._aggregate_sum
        elif method_str == "ave":
            return self._aggregate_ave
        elif method_str == "max":
            return self._aggregate_max
        elif method_str == "min":
            return self._aggregate_min
        elif method_str == "count":
            return self._aggregate_count
        elif method_str == "std":
            return self._aggregate_std
    
    # 将data[index][name]对应得到的数据集放入聚合方法中，运算得到值
    def _get_aggregate_val(self, index, name, method_fun):
        if len(index):
            data = [int(self.data[i][self.columns.index(name)]) for i in index]
            return method_fun(data)
        return ""

    # 计算聚合表的表头，顺便计算了每个行和列对应的index的划分结果
    def _get_print_table_header(self):
        self.print_table = [["" for i in range(1000)] for j in range(1000)]
        self.print_table_size = [0,0]

        self.row_index_data = []
        self.col_index_data = []

        # 输出列表头
        start_col_index = self.row_leaves[0].depth + 1 + (self.aggregate_position&1)*(self.aggregate_method_num-1) #为多个值的行标记预留空间
        start_row_index = 0 
        col_seg = max(1, (self.aggregate_position^1)*self.aggregate_method_num) 

        def print_col_dfs(node,col_ind):
            row_ind = start_row_index + node.depth
            self.print_table[row_ind][col_ind] = node.val if isinstance(node.val, str) else str(node.val)
            if len(node.son) == 0:
                if self.aggregate_position^1 and self.aggregate_method_num-1:
                    for j in range(self.aggregate_method_num):
                        self.print_table[row_ind+1][col_ind+j] = self.aggregate_method[j] + ":" + self.aggregate_name[j] 
                        self.col_index_data.append(node.data_index)
                else:
                    self.col_index_data.append(node.data_index)

            for son in node.son:
                col_ind = col_seg+print_col_dfs(son, col_ind)
            if len(node.son):   #非叶子节点加入汇总行
                if self.aggregate_position^1: # 如果值在列处
                    for i in range(self.aggregate_method_num):
                        self.print_table[row_ind][col_ind+i] = "汇总 " + self.aggregate_method[i] + ":"+ (node.val if isinstance(node.val, str) else str(node.val))
                        self.col_index_data.append(node.data_index)
                else:   #如果值不在列处
                    self.print_table[row_ind][col_ind] = "汇总:" + (node.val if isinstance(node.val, str) else str(node.val))
                    self.col_index_data.append(node.data_index)

            return col_ind
        
        self.print_table_size[1] = max(self.print_table_size[1], print_col_dfs(self.col_root, start_col_index)+col_seg) 
        
        # 输出行表头
        start_col_index = 0 
        start_row_index = self.col_leaves[0].depth + 1 + (self.aggregate_position^1)*(self.aggregate_method_num-1) #为多个值的行标记预留空间
        row_seg = max(1, (self.aggregate_position&1)*self.aggregate_method_num)

        def print_row_dfs(node,row_ind):
            col_ind = start_col_index + node.depth

            self.print_table[row_ind][col_ind] = node.val if isinstance(node.val, str) else str(node.val)
            
            if len(node.son) == 0:
                if self.aggregate_position&1 and self.aggregate_method_num-1:
                    for j in range(self.aggregate_method_num):
                        self.print_table[row_ind+j][col_ind+1] = self.aggregate_method[j] + ":" + self.aggregate_name[j] 
                        self.row_index_data.append(node.data_index)
                else:
                    self.row_index_data.append(node.data_index)

            for son in node.son:
                row_ind = row_seg+print_row_dfs(son, row_ind)
            
            if len(node.son):   #非叶子节点加入汇总行
                if self.aggregate_position&1: # 如果值在行处
                    for i in range(self.aggregate_method_num):
                        self.print_table[row_ind+i][col_ind] = "汇总 " + self.aggregate_method[i] + ":"+ (node.val if isinstance(node.val, str) else str(node.val))
                        self.row_index_data.append(node.data_index)
                else: # 如果值不在行处
                    self.print_table[row_ind][col_ind] = "汇总:" + (node.val if isinstance(node.val, str) else str(node.val))
                    self.row_index_data.append(node.data_index)
            return row_ind
        
        self.print_table_size[0] = max(self.print_table_size[0], print_row_dfs(self.row_root, start_row_index)+row_seg) 
        

    # 根据index的划分结果，计算出聚合表中每个位置的聚合值，并储存
    def _get_print_table_val(self):
        start_col_index = self.row_leaves[0].depth + 1 + (self.aggregate_position&1)*(self.aggregate_method_num-1) #为多个值的行标记预留空间
        start_row_index = self.col_leaves[0].depth + 1 + (self.aggregate_position^1)*(self.aggregate_method_num-1)  #为多个值的行标记预留空间

        for i in range(len(self.row_index_data)):
            for j in range(len(self.col_index_data)):
                x = i+start_row_index
                y = j+start_col_index

                # 这里是如果有多个聚合方法，需要判断使用哪个聚合方法
                if self.aggregate_position^1: # 如果值在列处
                    method_choose = j % self.aggregate_method_num
                else:   # 如果值在行处
                    method_choose = i % self.aggregate_method_num

                data_index = self.row_index_data[i] & self.col_index_data[j]   # 需要将行列的index相交，以得到对应单元格的数据
                self.print_table[x][y] = self._get_aggregate_val(
                    data_index, 
                    self.aggregate_name[method_choose],
                    self._get_aggregate_method(self.aggregate_method[method_choose]))
                
                self.print_table[x][y] = str(self.print_table[x][y])    #转换成字符串，方便输出
            
    # 输出表格，可以把注释那段打开，然后把另外一段注释掉，不过可能会导致输出太长，不建议使用
    def print(self):
        for row in self.print_table[:self.print_table_size[0]]:
            # print("\t\t".join(row[:self.print_table_size[1]]))
            print(",".join(row[:self.print_table_size[1]]))

    # 将表储存在pivot_table.csv文件中，建议使用
    def save(self):
        with open("pivot_table.csv",'w') as f:
            for row in self.print_table[:self.print_table_size[0]]:
                f.write(",".join(row[:self.print_table_size[1]]))
                f.write("\n")



if __name__ == "__main__":
    # 数据读入部分
    try:
        f = open(r'pypivot.csv', 'r', encoding='utf-8')
        file_content = f.read().strip('\ufeff').rstrip('\n')
        data = []

        lines = file_content.split('\n')
        for line in lines:
            row = line.split(',')
            data.append(row)
    except FileNotFoundError:
        print("The specified file could not be found.")
    data = [val[1:] for val in data]
    columns_name = data[0]
    data = data[1:]

    # 这是一个可用的示例 可以把注释行打开来尝试不同效果
    table = pivot_table(data,columns_name)
    table.add_col("Employment")
    table.add_row("Gender")
    # table.add_col("Age")
    table.add_row("Age")
    table.add_agg("Salary","sum")
    table.add_agg("Salary","ave")
    # table.del_agg("Salary","ave")

    table.save()
    table.print()
            