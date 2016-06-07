# coding: utf-8

import lupa

lua = lupa.LuaRuntime()
lua.execute("dofile 'lua.lua' ")
method = self.lua.globals().method
method('path')