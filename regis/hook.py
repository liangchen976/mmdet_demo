import sys
class HOOK:

    def before_breakfirst(self, runner):
        print('{}:吃早饭之前晨练30分钟'.format(sys._getframe().f_code.co_name))

    def after_breakfirst(self, runner):
        print('{}:吃早饭之前晨练30分钟'.format(sys._getframe().f_code.co_name))

    def before_lunch(self, runner):
        print('{}:吃午饭之前跑上实验'.format(sys._getframe().f_code.co_name))

    def after_lunch(self, runner):
        print('{}:吃完午饭午休30分钟'.format(sys._getframe().f_code.co_name))

    def before_dinner(self, runner):
        print('{}: 没想好做什么'.format(sys._getframe().f_code.co_name))

    def after_dinner(self, runner):
        print('{}: 没想好做什么'.format(sys._getframe().f_code.co_name))

    def after_finish_work(self, runner, are_you_busy=False):
        if are_you_busy:
            print('{}:今天事贼多，还是加班吧'.format(sys._getframe().f_code.co_name))
        else:
            print('{}:今天没啥事，去锻炼30分钟'.format(sys._getframe().f_code.co_name))


class Runner(object):
    def __init__(self, ):
        pass
        self._hooks = []

    def register_hook(self, hook):
        # 这里不做优先级判断，直接在头部插入HOOK
        print('inner', hook)
        self._hooks.insert(0, hook)
        print(len(self._hooks))

    def call_hook(self, hook_name):
        for hook in self._hooks:
            getattr(hook, hook_name)(self)

    def run(self):
        print('开始启动我的一天')
        self.call_hook('before_breakfirst')
        self.call_hook('after_breakfirst')
        self.call_hook('before_lunch')
        self.call_hook('after_lunch')
        self.call_hook('before_dinner')
        self.call_hook('after_dinner')
        self.call_hook('after_finish_work')
        print('~~睡觉~~')

if __name__ == '__main__':
    runner = Runner()
    hook = HOOK()
    print(hook)
    runner.register_hook(hook)
    runner.run()