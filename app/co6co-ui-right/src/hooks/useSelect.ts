

import { ref } from 'vue';
import { ISelect, IEnumSelect, IResponse } from 'co6co';
import { getTagTypeStr } from '@/utils/other';
import { IEnumsResonse } from '@/constants/api';
export const useSelect = <T extends any = void>(api: (data: T) => Promise<IResponse<ISelect[]>>) => {
    const selectData = ref<ISelect[]>([]);
    // 使用条件类型定义 refresh 函数 
    const refresh = async (data: T) => {
        selectData.value = [];
        const res = await api(data);
        selectData.value = res.data;
    };
    const getName = (value?: number) => {
        if (value != undefined) return selectData.value.find((m) => m.id == value)?.name;
        return '';
    };
    const loadData = refresh;
    return { loadData, selectData, refresh, getName };
}

// , nameChecked?: (item: IEnumSelect) => boolean 不该把nameChecked 提出来，因为 value 变化的,赋值后 m=>m.value==1 固定了不能再变化
export const useEnumSelect = <T extends any = void>(api: (data: T) => Promise<IResponse<IEnumSelect[]>>) => {
    const selectData = ref<IEnumSelect[]>([]);
    const refresh = async (data: T) => {
        selectData.value = [];
        const res = await api(data);
        if (res.data instanceof Array) {
            selectData.value = res.data;
        }
    };

    const getName = (value?: number | string) => {
        const nameChecked = (m) => m.value == value;
        return selectData.value.find(nameChecked!)?.label; 
    };

    const getTagType = (value?: number | string, checked?: (item: IEnumSelect) => boolean) => {
        if (!checked) checked = (m) => m.value == value;
        return getTagTypeStr(selectData.value, checked!);
    };
    const loadData = refresh;
    return { loadData, selectData, refresh, getName, getTagType };
};

export const useEnumsSelect = <T extends any = void>(api: (data: T) => Promise<IResponse<IEnumsResonse>>) => {
    const selectData = ref<{ [key: string]: IEnumSelect[] }>();
    const refresh = async (data: T) => {
        selectData.value = {};
        const res = await api(data);
        selectData.value = res.data;
    };

    const getName = (key: string, value?: number | string, nameChecked?: (item: IEnumSelect) => boolean) => {
        if (value == undefined) {
            if (!nameChecked) nameChecked = (m) => m.value == value;
            if (selectData.value && key in selectData.value)
                return selectData.value[key].find(nameChecked)?.label;
        }
        return '';
    };

    const getTagType = (key: string, value?: number | string, checked?: (item: IEnumSelect) => boolean) => {
        if (!checked) checked = (m) => m.value == value;
        if (selectData.value && key in selectData.value)
            return getTagTypeStr(selectData.value[key], checked!);
    };
    const loadData = refresh;
    return { loadData, selectData, refresh, getName, getTagType };
};