"""
演示字典的构造
注意，即便遍历字典时，键—值对的返回顺序也与存储顺序不同。Python不关心键—值对的存储顺序，而只跟踪键和值之间的关联关系。
"""
user_0 = {
    'username':'efermi',
    'first':'enrico',
    'last':'fermi',
}


for key, value in user_0.items():
    print("\nKey: " + key)
    print("value: " + value)

favorite_languages = {
    'jen':'python',
    'sarah':'c',
    'edward':'ruby',
    'phil':'python'
    }

#处的代码让Python遍历字典中的每个键—值对，并将键存储在变量name 中，而将值存储在变量language 中
for name, language in favorite_languages.items():
    print(name.title() + "'s favorite language is " + language.title() + ".")

#遍历字典中的所有键
for name in favorite_languages.keys():
    print(name.title())

for language in favorite_languages.values():
    print(language.title())