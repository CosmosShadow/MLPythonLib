# coding: utf-8

import lupa

lua = lupa.LuaRuntime()
lua.execute("dofile 'lua.lua' ")
method = lua.globals().method
method('path')

a = lua.globals().a

print sorted(a.values())
print a
print len(a)
for i in range(len(a)):
	print a[i+1]
