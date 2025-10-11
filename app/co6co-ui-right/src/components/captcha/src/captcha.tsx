
import { defineComponent, ref,  onMounted,  } from 'vue';
import { create_URL_resource } from '@/api/download';
import { get_captcha_img, verify_captcha } from '@/api/verify'; 
import {   ElInput, ElMessage } from 'element-plus'

import { useModelWrapper } from 'co6co';
const Captcha = defineComponent({
	props: {
		placeholder: {
			type: String,
			default: '请输入验证码',
		},
		modelValue: {
			type: String,
		},
	},
	emits: {
		'update:modelValue': (data: String) => true,
		'verified': () => true,
	},
	name: 'Captcha',
	setup(props, ctx) {


		const captchaImage = ref('')
		const userInput = ref('')
		const errorMessage = ref('')
		// 获取验证码图片
		const fetchCaptcha = async () => {
			try {
				const res = await get_captcha_img()
				captchaImage.value = create_URL_resource({ data: res.data })
			} catch (error) {
				console.error('获取验证码失败:', error)
			}
		}
		// 刷新验证码
		const refreshCaptcha = () => {
			fetchCaptcha()
			errorMessage.value = '' // 清除错误信息
			userInput.value = '' // 清空输入框
		}
		// 验证验证码
		const verifyCaptcha = async () => {
			if (!userInput.value.trim()) {
				errorMessage.value = '请输入验证码'
				return
			}
			try {
				verify_captcha({ code: userInput.value })
				.then((res) => {
					if (res.code == 0) {
						ctx.emit('verified') // 验证成功，通知父组件
						ElMessage.success('验证成功')
					} else {
						ElMessage.error('验证失败，请重试' + res.message)
						refreshCaptcha()
					}
				})
				.catch((err) => {
					ElMessage.error('验证失败，请重试')
					refreshCaptcha()
				})
			} catch (error) {
				errorMessage.value = '验证失败，请稍后再试'
			}
		}
		onMounted(() => {
			fetchCaptcha()
		})
		const { localValue, onChange } = useModelWrapper(props, ctx);
		const rander = () => (
			<>
				<ElInput v-model={localValue.value} placeholder="请输入验证码" onChange={onChange}>
					{{
						prepend: () => (
							<img
								class="el-input__inner"
								src={captchaImage.value}
								style="cursor: pointer;"
								onClick={refreshCaptcha}
							/>
						)
					}}
				</ElInput>
			</>
		);
		//真是方法
		ctx.expose({
			verifyCaptcha,
			refreshCaptcha,
		});
		//.d.ts 中的定义
		rander.verifyCaptcha = verifyCaptcha;
		rander.refreshCaptcha = refreshCaptcha;
		return rander;
	},
});

export default Captcha;
