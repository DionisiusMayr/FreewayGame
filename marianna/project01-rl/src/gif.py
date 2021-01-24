from matplotlib import animation
import matplotlib.pyplot as plt
import src.environment as environment

# From https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
def _save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    
def generate_gif(fn: str, agent, reduce_state, RAM_mask):
    env, initial_state = environment.get_env()
    game_over = False
    state = reduce_state(initial_state)[RAM_mask].data.tobytes()  # Select useful bytes
    action = agent.act(state)

    frames = []
    FRAME_FREQ = 2

    for t in range(50000):
        if t % FRAME_FREQ == 0:
            frames.append(env.render(mode="rgb_array"))

        ob, _, game_over, _ = env.step(action)

        ob = reduce_state(ob)
        state = ob[RAM_mask].data.tobytes()
        action = agent.act(state)  # Next action

        if game_over:
            break

    _save_frames_as_gif(frames=frames, path='./gif/', filename=f'{fn}.gif')