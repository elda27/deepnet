import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import pytest
import auto_path
import config

@pytest.mark.parametrize(('config_string', 'check_params'), [
('''
[test1]
mark=1
[test2]
mark="${test1.mark}"
[test3]
mark="value/${test1.mark}"
''', { 'test1.mark':1, 'test2.mark': 1, 'test3.mark': 'value/1'} 
),
('''
[test1]
mark='value'
[test2]
mark="${test1.mark}"
[test3]
mark="${test1.mark}/test"
''', { 'test1.mark':'value', 'test2.mark': 'value', 'test3.mark': 'value/test'} 
),
])
def test_decoder(config_string, check_params):
    param = config.loads(config_string)
    for nest_key, answer in check_params.items():
        nest_keys = nest_key.split('.')
        temp_dict = param
        for key in nest_keys:
            temp_dict = temp_dict[key]
        
        assert temp_dict == answer, 'Incorrect value (Key:{nest_key})'.format(nest_key=nest_key)

@pytest.mark.parametrize(('config_string', 'check_params'), [
('''
[test1]
mark = [ 1, 2, 3 ]
[test2]
mark = "value/${test1.mark}"
''', { 'test1.mark': [1, 2, 3], 'test2.mark': [ 'value/1', 'value/2', 'value/3' ]} 
),
('''
[test1]
mark=[ 'value1', 'value2', 'value3' ]
[test2]
mark="${test1.mark}"
[test3]
mark="${test1.mark}/test"
''', 
    {
        'test1.mark': [ 'value1', 'value2', 'value3' ],
        'test2.mark': [ 'value1', 'value2', 'value3' ], 
        'test3.mark': [ 'value1/test', 'value2/test', 'value3/test' ]
    } 
),
('''
[test1]
mark=[ 'value1', 'value2', 'value3' ]
value='value'
[test2]
mark="${test1.mark}"
[test3]
mark="${test1.mark}/test"
[test4]
mark="${test1.value}/${test1.mark}"
''', 
    {
        'test1.mark': [ 'value1', 'value2', 'value3' ],
        'test2.mark': [ 'value1', 'value2', 'value3' ], 
        'test3.mark': [ 'value1/test', 'value2/test', 'value3/test' ],
        'test4.mark': [ 'value/value1', 'value/value2', 'value/value3' ]
    } 
),
])
def test_decoder_list_expansion(config_string, check_params):
    param = config.loads(config_string)
    for nest_key, answer in check_params.items():
        nest_keys = nest_key.split('.')
        temp_dict = param
        for key in nest_keys:
            temp_dict = temp_dict[key]
        
        if isinstance(answer, list):
            for i, (val, ans) in enumerate(zip(temp_dict, answer)):
                assert val == ans, \
                  'Incorrect value (Key:{nest_key}[{index}], {list})'.format(nest_key=nest_key, index=i, list=temp_dict)
        else:
            assert temp_dict == answer, 'Incorrect value (Key:{nest_key})'.format(nest_key=nest_key)

if __name__ == '__main__':
    pytest.main()