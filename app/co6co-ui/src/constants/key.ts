// eslint-disable-next-line prettier/prettier
export const INSTALLED_KEY: unique symbol = Symbol('INSTALLED_CO6CO_KEY')
const PERMISS_KEY = Symbol('PERMISS_KEY')
export const ConstObject = {
  [PERMISS_KEY]: 'permiss',
  getPermissValue: () => {
    return ConstObject[PERMISS_KEY]
  },
}
