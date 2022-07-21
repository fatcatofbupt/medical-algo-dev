import yaml, json
task_name = 'fundus_create_obs'
with open('../' + task_name + '.yaml','r') as f:
    f_dict = yaml.load(f)
out_json = json.dumps(f_dict,sort_keys=False,indent=4,separators=(',', ': '))
with open(task_name + '.json','w') as f:
    # json.dump(f_dict,f,sort_keys=False,indent=4,separators=(',', ': '))
    json.dump(f_dict,f)
print(out_json)
