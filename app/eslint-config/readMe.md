 # .prettierrc
 {
 "endOfLine": "auto"    //不检测文件每行结束的格式
 //  git config --global core.autocrlf false
 //- 设置true时，在拉取代码换行(LF)会被转换成回车和换行(CRLF)。
   - 设置false时 提交和拉取都将不做转换操作,保留原有版本。
   - 设置input时，提交会把回车和换行转(CRLF)换成换行(LF)，拉取时不转换。
 }

 # .eslintrc.json
 ```
 prettier/prettier": "error"  //来将Prettier的规则集成到ESLint中
 ```