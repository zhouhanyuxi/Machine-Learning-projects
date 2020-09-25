import os
import sys
import random

"""
对模型返回结果进行评价
"""


class evaluator:

    def __init__(self, full_set_path, test_set_path):
        """evaluator 构造器

            构造一个评价器

            Args:
                :param full_set_path: 全集的路径
                :param test_set_path: 测试集的路径
            """
        self.dataset = full_set_path
        self.test = test_set_path
        self.label_map = self.__load_labels()

    def get_random_test_list(self, num: int) -> list:
        """从测试集获取一个随机的样本list

            在测试集的所有样本中随机截取一段数量等于 num 的图片路径用于evaluation

            Args:
                :param num: 截取图片的个数 如果num小于等于0则视为取全集进行评价

            Returns:
                :return list[path]
                返回一个包含图片路径的列表

            Raises:

            """
        all_pic = []
        for label in os.listdir(self.test):
            all_pic.extend([label + '/' + x for x in os.listdir(os.path.join(self.test, label))])

        if num <= 0:
            return all_pic

        else:
            random.shuffle(all_pic)
            position = random.randint(0, (len(all_pic) - num))
            return all_pic[position:(position + num)]

    def get_precision(self, input_image: str, result_list: list, k: list) -> list:
        """获取precision@k

                计算precision的值

                Args:
                    :param k: k值列表(precision@k)
                    :param result_list: 模型返回结果的列表
                    :param input_image: 输入模型的测试图片路径

                Returns:
                    :return list[
                        precision@k[0](all),
                        precision@k[0](test),
                        precision@k[1](all),
                        precision@k[1](test),
                        ...
                        precision@k[len(k)-1](all),
                        precision@k[len(k)-1](test),
                    ]
                    返回一个包含图片路径的列表

                Raises:
                    :exception Invalid Result Length Exception: k列表中的k值大于返回结果的数目
                    :exception Invalid Path Exception:  路径错误
                    :exception Invalid k value Exception: 错误的k值 (k<=0)
                """
        self.__do_check_parameter(input_image, result_list, k, 'precision')
        out_put_result_all = []
        out_put_result_test = []
        trigger = 1
        count_all = 0
        count_test = 0
        for result in result_list:
            if self.__is_same_label(input_image, result):
                count_all += 1
                if self.__is_test(result):
                    count_test += 1
                else:
                    pass
            else:
                pass

            if trigger in k:
                result_all = count_all / trigger
                result_test = count_test / trigger
                # print('precision@{}(all)={}'.format(trigger, result_all))
                # print('precision@{}(test)={}'.format(trigger, result_test))
                out_put_result_all.append(result_all)
                out_put_result_test.append(result_test)
            trigger += 1
        
        return out_put_result_all,out_put_result_test

    def get_recall(self, input_image: str, result_list: list, r: list) -> list:
        """获取recall@r

                        计算recall的值

                        Args:
                            :param r: r值列表(recall@r)
                            :param result_list: 模型返回结果的列表
                            :param input_image: 输入模型的测试图片路径

                        Returns:
                            :return list[
                                recall@r[0](all),
                                recall@r[0](test),
                                recall@r[1](all),
                                recall@r[1](test),
                                ...
                                recall@r[len(r)-1](all),
                                recall@r[len(r)-1](test),
                            ]
                            返回一个包含图片路径的列表

                        Raises:
                            :exception Invalid Result Length Exception: k列表中的k值大于返回结果的数目
                            :exception Invalid Path Exception:  路径错误
                            :exception Invalid k value Exception: 错误的k值 (k<=0)
                        """
        self.__do_check_parameter(input_image, result_list, r, 'recall')
        out_put_result_all = []
        out_put_result_test = []
        trigger = 1
        temp = input_image.split('/')
        label = temp[len(temp) - 2]
        base_all = self.__get_label_len(1, label)
        base_test = self.__get_label_len(0, label)
        count_all = 0
        count_test = 0
        for result in result_list:
            if self.__is_same_label(input_image, result):
                count_all += 1
                if self.__is_test(result):
                    count_test += 1
                else:
                    pass
            else:
                pass

            if trigger in r:
                result_all = count_all / base_all
                result_test = count_test / base_test
                #print('recall@{}(all)={}'.format(trigger, result_all))
                #print('recall@{}(test)={}'.format(trigger, result_test))
                out_put_result_all.append(result_all)
                out_put_result_test.append(result_test)
                
            trigger += 1

        return out_put_result_all,out_put_result_test

    @staticmethod
    def __do_check_parameter(input_image: str, result_list: list, p: list, p_r: str):

        for i in p:
            if i > len(result_list):
                try:
                    raise ProcessException('Invalid Result Length Exception: ',
                                           '{}@{} need at least{} results '
                                           'but get{}.'.format(p_r, i, i, len(result_list)))
                except ProcessException as e:
                    print(e.name, e.reason)
                    sys.exit(0)

            if i <= 0:
                try:
                    raise ProcessException('Invalid k value Exception: ',
                                           '{}@{} is not available. '.format(p_r, i))
                except ProcessException as e:
                    print(e.name, e.reason)
                    sys.exit(0)

    #    if not os.path.isdir(input_image):
    #        try:
    #            raise ProcessException('Invalid Path Exception: ',
    #                                   '{} is not a valid path'.format(input_image))
    #        except ProcessException as e:
    #            print(e.name, e.reason)
    #            sys.exit(0)

    def __is_test(self, image):
        temp = image.split('/')
        name = temp[len(temp) - 1]
        label = temp[len(temp) - 2]
        images_of_label = os.listdir(os.path.join(self.test, label))
        return name in images_of_label

    def __is_same_label(self, image1, image2):
        return self.__get_label(image1) == self.__get_label(image2)

    def __get_label(self, image: str) -> int:
        temp = image.split('/')
        label = self.label_map[temp[len(temp) - 2]]
        return label

    def __get_label_len(self, trigger, label):
        if trigger == 1: 
            return len(os.listdir(os.path.join(self.dataset, label)))
        else:
            return len(os.listdir(os.path.join(self.test, label)))

    def __load_labels(self) -> {}:
        label_map = {}
        label_list = os.listdir(self.dataset)
        index = 0
        for label in label_list:
            label_map[label] = index
            index += 1
        return label_map


class ProcessException(Exception):
    def __init__(self, name, reason):
        self.name = name
        self.reason = reason
