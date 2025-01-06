import numpy as np
import pandas as pd
import random


class WhaleOptimizer:
    def __init__(self, data_file, max_iter=3, whale_num=3):
        self.data = pd.read_excel(data_file, sheet_name='修改后数据')
        self.max_iter = max_iter
        self.whale_num = whale_num
        self.dim = self.data.shape[0]  # 每个变量对应一个单品销量
        self.LB = np.zeros(self.dim)  # 下界设为0
        self.UB = np.full(self.dim, 3)
        self.X = np.random.uniform(0, 1, (whale_num, self.dim)) * (self.UB - self.LB) + self.LB
        self.X = np.round(self.X).astype(int)  # 将 X 取整并转换为整数类型
        self.V = np.zeros((whale_num, self.dim))
        self.gBest_score = 0  # 初始蛋白质氨基酸评分为0
        self.gBest_X = self.X[0, :].copy()  # 初始化为随机一个鲸鱼的位置



    def fitFunc(self, input):
        edible_quantity = np.array(input)
        protein = np.array(self.data['蛋白质g'])

        # 将数据分成早、中、晚三组
        breakfast_indices = range(33)
        lunch_indices = range(33, 92)
        dinner_indices = range(92, 141)

        # 计算每组的得分
        breakfast_score = self.calculate_score(edible_quantity, protein, breakfast_indices)
        lunch_score = self.calculate_score(edible_quantity, protein, lunch_indices)
        dinner_score = self.calculate_score(edible_quantity, protein, dinner_indices)

        # 将得分相加得到总分
        total_score = (breakfast_score + lunch_score + dinner_score)/3

        return total_score

    def calculate_score(self, edible_quantity, protein, indices):
        amino_acid_1 = sum(np.array(self.data['异亮氨酸'])[indices] * edible_quantity[indices])
        amino_acid_2 = sum(np.array(self.data['亮氨酸'])[indices] * edible_quantity[indices])
        amino_acid_3 = sum(np.array(self.data['赖氨酸'])[indices] * edible_quantity[indices])
        amino_acid_4 = sum(np.array(self.data['含硫氨基酸'])[indices] * edible_quantity[indices])
        amino_acid_5 = sum(np.array(self.data['芳香族氨基酸'])[indices] * edible_quantity[indices])
        amino_acid_6 = sum(np.array(self.data['苏氨酸'])[indices] * edible_quantity[indices])
        amino_acid_7 = sum(np.array(self.data['色氨酸'])[indices] * edible_quantity[indices])
        amino_acid_8 = sum(np.array(self.data['缬氨酸'])[indices] * edible_quantity[indices])

        meal_protein = sum(edible_quantity[indices] * protein[indices])

        if meal_protein == 0:
            return -np.inf  # 避免除零错误


        scores = np.array([
            amino_acid_1 / (meal_protein * 40 * 0.01),
            amino_acid_2 / (meal_protein * 70 * 0.01),
            amino_acid_3 / (meal_protein * 55 * 0.01),
            amino_acid_4 / (meal_protein * 35 * 0.01),
            amino_acid_5 / (meal_protein * 60 * 0.01),
            amino_acid_6 / (meal_protein * 40 * 0.01),
            amino_acid_7 / (meal_protein * 10 * 0.01),
            amino_acid_8 / (meal_protein * 50 * 0.01)
        ])

        score = np.min(scores)
        return score

    def apply_constraints(self,):
        energy = np.array(self.data['能量'])
        protein = np.array(self.data['蛋白质g'])
        fat = np.array(self.data['脂肪g'])
        carbohydrate = np.array(self.data['碳水化合物g'])
        boy_nonproductive_nutrient = self.data.loc[:, '钙mg':'维生素Cmg'].values
        standard_boy_nonproductive_nutrient = [800,12,12.5,800,1.4,1.4,100]

        perfect_matrix = []


        while len(perfect_matrix) < self.whale_num:
            X = self.X

            for i in range(self.whale_num):
                num_active_vars = np.random.randint(9, 13)
                active_indices = np.random.choice(self.dim, num_active_vars, replace=False)
                inactive_indices = np.setdiff1d(np.arange(self.dim), active_indices)
                X[i, inactive_indices] = 0

            for i in range(self.whale_num):
                total_energy = np.sum(X[i] * energy)
                while total_energy > 2640:
                    non_zero_indices = np.where(X[i] > 0)[0]
                    if len(non_zero_indices) == 0:
                        break
                    idx = np.random.choice(non_zero_indices)
                    X[i, idx] = max(X[i, idx] - 1, 0)
                    total_energy = np.sum(X[i] * energy)

                while total_energy <= 2160:
                    idx = np.random.choice(self.dim)
                    X[i, idx] = min(X[i, idx] + 1, self.UB[idx])
                    total_energy = np.sum(X[i] * energy)


                protein_energy_ratio = np.sum(X[i] * protein * 4) / total_energy
                fat_energy_ratio = np.sum(X[i] * fat * 9) / total_energy
                carbohydrate_energy_ratio = np.sum(X[i] * carbohydrate * 4) / total_energy

                if 0.10 <= protein_energy_ratio <= 0.15 and 0.20 <= fat_energy_ratio <= 0.30 and 0.50 <= carbohydrate_energy_ratio <= 0.65:
                    # print("protein_energy_ratio",protein_energy_ratio)
                    # print("fat_energy_ratio",fat_energy_ratio)
                    # print("carbohydrate_energy_ratio",carbohydrate_energy_ratio)
                    nutrient_totals = np.sum(X[:, :, np.newaxis] * boy_nonproductive_nutrient, axis=1)
                    deviation = nutrient_totals - standard_boy_nonproductive_nutrient
                    uniform_deviation = deviation/standard_boy_nonproductive_nutrient
                    # print("筛选前",uniform_deviation)
                    filtered_matrices = uniform_deviation[np.all(np.abs(uniform_deviation) < 0.5, axis=1)]

                    if filtered_matrices.size > 0:
                        energy_breakfast = (np.sum(X[:, :33] * energy[:33], axis=1))/total_energy
                        energy_lunch = (np.sum(X[:, 33:92] * energy[33:92], axis=1))/total_energy
                        energy_dinner = (np.sum(X[:, 92:] * energy[92:], axis=1))/total_energy


                        if (np.all((0.25 <= energy_breakfast[i]) & (energy_breakfast[i] <= 0.35)) and
                                np.all((0.30 <= energy_lunch[i]) & (energy_lunch[i] <= 0.40)) and
                                np.all((0.30 <= energy_dinner[i]) & (energy_dinner[i] <= 0.40))):
                            perfect_matrix.append(X[i])
                            print(len(perfect_matrix))
                            continue
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
        return np.array(perfect_matrix)




    def opt(self):
        t = 0
        while t < self.max_iter:
            self.X = self.apply_constraints()  # Apply constraints before evaluation
            for i in range(self.whale_num):
                self.X[i, :] = np.clip(self.X[i, :], self.LB, self.UB)  # 检查边界
                fit = self.fitFunc(self.X[i, :])
                # 更新全局最优解
                if fit > self.gBest_score and check_constraints(self,self.X[i, :]):
                    self.gBest_score = fit
                    self.gBest_X = self.X[i, :].copy()
                print(self.gBest_X)

            a = 2 * (self.max_iter - t) / self.max_iter
            for i in range(self.whale_num):
                p = np.random.uniform()
                R1 = np.random.uniform()
                R2 = np.random.uniform()
                A = 2 * a * R1 - a
                C = 2 * R2
                l = 2 * np.random.uniform() - 1

                if p >= 0.5:
                    D = abs(self.gBest_X - self.X[i, :])
                    self.V[i, :] += A * D
                else:
                    if abs(A) < 1:
                        D = abs(C * self.gBest_X - self.X[i, :])
                        self.V[i, :] += D
                    else:
                        rand_index = np.random.randint(low=0, high=self.whale_num)
                        X_rand = self.X[rand_index, :]
                        D = abs(X_rand - self.X[i, :])
                        self.V[i, :] += D

                # 将 V 取整并强制为整数类型
                self.V[i, :] = np.round(self.V[i, :]).astype(int)
                # 更新位置后将 X 强制为整数类型
                self.X[i, :] = self.X[i, :].astype(int) + self.V[i, :]
                self.X[i, :] = np.clip(self.X[i, :], self.LB, self.UB)

                # 检查并更新全局最优解

                fit = self.fitFunc(self.X[i, :])
                if fit > self.gBest_score and check_constraints(self,self.X[i, :]):
                    self.gBest_score = fit
                    self.gBest_X = self.X[i, :].copy()
                print(self.gBest_X)

            if t % 10 == 0:
                print(f'At iteration: {t}, Best Score: {self.gBest_score}')
            t += 1

        return self.gBest_score, self.gBest_X

def check_constraints(self,gBest_X):
    energy = np.array(self.data['能量'])
    protein = np.array(self.data['蛋白质g'])
    fat = np.array(self.data['脂肪g'])
    carbohydrate = np.array(self.data['碳水化合物g'])
    boy_nonproductive_nutrient = self.data.loc[:, '钙mg':'维生素Cmg'].values
    standard_boy_nonproductive_nutrient = [800, 12, 12.5, 800, 1.4, 1.4, 100]

    total_energy = np.sum(gBest_X * energy)
    protein_energy_ratio = np.sum(gBest_X * protein * 4) / total_energy
    fat_energy_ratio = np.sum(gBest_X * fat * 9) / total_energy
    carbohydrate_energy_ratio = np.sum(gBest_X * carbohydrate * 4) / total_energy

    energy_breakfast = (np.sum(gBest_X[:33] * energy[:33])) / total_energy
    energy_lunch = (np.sum(gBest_X[33:92] * energy[33:92])) / total_energy
    energy_dinner = (np.sum(gBest_X[92:] * energy[92:])) / total_energy

    nutrient_totals = np.sum(gBest_X[:, np.newaxis] * boy_nonproductive_nutrient, axis=0)
    deviation = nutrient_totals - standard_boy_nonproductive_nutrient
    uniform_deviation = deviation / standard_boy_nonproductive_nutrient

    if (2160 <= total_energy <= 2640 and
               0.10 <= protein_energy_ratio <= 0.15 and
               0.20 <= fat_energy_ratio <= 0.30 and
               0.50 <= carbohydrate_energy_ratio <= 0.65 and
               np.all(abs(uniform_deviation) < 0.5) and
                0.25 <= energy_breakfast <= 0.35 and
                0.30 <= energy_lunch <= 0.40 and
                0.30 <= energy_dinner <= 0.40):
        return True
    else:
        return False





# 修改参数以适应新场景
data_file = "优化模型数据.xlsx"
optimizer = WhaleOptimizer(data_file)
gBest_score, gBest_X = optimizer.opt()


print("最大AAS评分", gBest_score)
print("最优解", gBest_X)


