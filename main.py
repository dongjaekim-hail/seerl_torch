import gym
from gym.wrappers import AtariPreprocessing
import torch
import torch.optim as optim
from model import A2C
from torch.nn import functional as F

def calculate_kl_divergence(policy1, policy2, state):
	action_probs1, _ = policy1(state)
	action_probs2, _ = policy2(state)

	kl_div = F.kl_div(action_probs1.log(), action_probs2, reduction='batchmean')
	return kl_div


# 정책 선택 과정에서의 목적 함수 계산
def compute_objective(policies, states, L_prime, weights, beta, Terr):
	B = []

	for i, policy1 in enumerate(policies):
		bi = []
		for state in states:
			# 정책 에러 L(s, a) 계산
			action_prob, _ = policy1(state)
			action = torch.argmax(action_prob).item()

			# L(s, a)의 임계값을 설정하여 잘못된 행동을 평가합니다.
			L_sa = 1 if abs(L_prime(state, action)) >= Terr else 0  # L_prime은 총 에러를 계산하는 함수

			# KL divergence 계산
			kl_div_sum = 0
			for j, policy2 in enumerate(policies):
				if i != j:
					kl_div_sum += calculate_kl_divergence(policy1, policy2, state).item()

			# B_i(s) 계산
			bi_value = L_sa - beta * (kl_div_sum / (len(policies) - 1))
			bi.append(bi_value)

		B.append(sum(bi))

	# P(s) 상태 확률, 여기서는 간단히 uniform 분포로 가정
	P_s = 1.0 / len(states)

	# 목적 함수 J(w) 계산
	J_w = P_s * torch.sum(torch.square(torch.sum(weights * torch.tensor(B))))

	return J_w


# 정책 선택 함수
def select_policies_via_optimization(policies, states, beta, Terr, num_policies_to_select):
	# 학습 가능한 파라미터 w 정의 (여기서는 초기값을 균일하게 설정)
	weights = torch.nn.Parameter(torch.ones(len(policies)) / len(policies), requires_grad=True)

	optimizer = torch.optim.Adam([weights], lr=0.01)

	for _ in range(100):  # 반복 횟수는 필요에 따라 조정
		optimizer.zero_grad()
		objective_value = compute_objective(policies, states, weights, beta)
		objective_value.backward()
		optimizer.step()

		# 가중치가 음수가 되지 않도록 클리핑
		with torch.no_grad():
			weights.clamp_(min=0)

	# 최종적으로 가장 큰 가중치를 가진 상위 정책 선택
	selected_indices = torch.argsort(weights, descending=True)[:num_policies_to_select]
	selected_policies = [policies[i] for i in selected_indices]

	return selected_policies


if __name__ == '__main__':
	env = gym.make('PongNoFrameskip-v4')
	env = AtariPreprocessing(env, screen_size=84, terminal_on_life_loss=True)
	input_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n
	alpha_0 = 1e-3
	T = int(1e5)
	M = int(5)
	beta = 1.2 # [1,2)

	# list of agents
	num_agents = 10

	agents = [A2C(input_dim, action_dim) for _ in range(num_agents)]
	optimizers = [optim.Adam(agent.parameters(), lr=alpha_0) for agent in agents]

	state, info = env.reset()
	state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

	done = False
	episodic_reward = 0

	for t in range(1, T+1):
		learning_rate = alpha_0 / 2 * (torch.cos(torch.tensor(torch.pi * (t-1) % (T//M) / (T//M))) + 1)
		for optimizer in optimizers:
			for param_group in optimizer.param_groups:
				param_group['lr'] = learning_rate

		action_ = []
		for model in agents:
			action_.append(model.select_action(state))
		# find the most common action
		action = torch.mode(torch.stack(action_)).values.item()

		next_state, reward, done, truncated, _ = env.step(action)
		next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
		episodic_reward += reward
		reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
		if done or truncated:
			done = torch.tensor(1, dtype=torch.float32).unsqueeze(0)
		done = torch.tensor(done, dtype=torch.float32).unsqueeze(0)


		loss_ = []
		for model in agents:
			loss = model.compute_loss(state, action, reward, next_state, done)
			model.update(optimizers[agents.index(model)], loss)
			loss_.append(loss.item())

		print(f't: {t}, loss: {sum(loss_)/num_agents}')

		state = next_state

		if done:
			print(f'Episode reward: {episodic_reward}')
			state, info = env.reset()
			state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
			done = False
			episodic_reward = 0


		if t % (T//M) == 0:
			# save agents
			for i, agent in enumerate(agents):
				torch.save(agent.state_dict(), f'agent_{i}.pt')

	# Policy selection process
	sample_states = [torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0) for _ in range(100)]
