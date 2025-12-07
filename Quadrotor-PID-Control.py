"""
================================================================================
SIMULADOR DE CONTROLE PID PARA QUADRICÓPTERO
================================================================================
Este código simula um drone quadricóptero controlado por PIDs em cascata.
Inclui modelo físico completo com arrasto aerodinâmico e visualização 3D.

Autor: Henrique de Albuquerque Vieira da Silva
Data: 2025
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk

# ============================================================================
# FUNÇÃO: SELEÇÃO DAS VARIÁVEIS
# ============================================================================
def abrir_configuracao():
    """
    Abre uma janela Tkinter organizada com LabelFrames e Abas.
    Retorna um dicionário com os valores escolhidos pelo usuário.
    """
    root = tk.Tk()
    root.title("Configuração da Simulação - Drone TCC")
    root.geometry("495x600") # Tamanho da tela inicial

    style = ttk.Style()
    style.theme_use('clam') 

    vars_dict = {}
    config_final = {}

    # Sistema de Abas
    tab_control = ttk.Notebook(root)
    tab_pid = ttk.Frame(tab_control)
    tab_voo = ttk.Frame(tab_control)
    parametros = ttk.Frame(tab_control)

    tab_control.add(tab_pid, text='Sintonia PID')
    tab_control.add(tab_voo, text='Cenário')
    tab_control.add(parametros, text='Parâmetros Físicos')
    tab_control.pack(expand=True, fill="both", padx=10, pady=(10, 5))

    # --- Função auxiliar ajustada para aceitar um 'container' (Frame) ---
    def criar_input(container, label_texto, chave, valor_padrao, linha):
        tk.Label(container, text=label_texto).grid(row=linha, column=0, padx=5, pady=5, sticky="e")
        var = tk.StringVar(value=str(valor_padrao))
        tk.Entry(container, textvariable=var, width=10).grid(row=linha, column=1, padx=5, pady=5)
        vars_dict[chave] = var

    # ================= ABA 1: SINTONIA PID =================
    # Criar quadros para agrupar visualmente
    frame_pos = tk.LabelFrame(tab_pid, text="Posição (X, Y)", font=("Arial", 9, "bold"), padx=10, pady=5)
    frame_pos.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

    frame_alt = tk.LabelFrame(tab_pid, text="Altitude (Z)", font=("Arial", 9, "bold"), padx=10, pady=5)
    frame_alt.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

    frame_att = tk.LabelFrame(tab_pid, text="Atitude (Roll/Pitch)", font=("Arial", 9, "bold"), padx=10, pady=5)
    frame_att.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

    frame_yaw = tk.LabelFrame(tab_pid, text="Atitude (Yaw)", font=("Arial", 9, "bold"), padx=10, pady=5)
    frame_yaw.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

    # Inputs Posição
    criar_input(frame_pos, "Kp:", "pid_pos_kp", 2.5, 0)
    criar_input(frame_pos, "Ki:", "pid_pos_ki", 0.7, 1)
    criar_input(frame_pos, "Kd:", "pid_pos_kd", 2.1, 2)

    # Inputs Altitude
    criar_input(frame_alt, "Kp:", "pid_z_kp", 2.0, 0)
    criar_input(frame_alt, "Ki:", "pid_z_ki", 0.5, 1)
    criar_input(frame_alt, "Kd:", "pid_z_kd", 2.5, 2)

    # Inputs Atitude Roll/Pitch
    criar_input(frame_att, "Kp:", "pid_att_kp", 2.0, 0)
    criar_input(frame_att, "Ki:", "pid_att_ki", 2.0, 1)
    criar_input(frame_att, "Kd:", "pid_att_kd", 0.5, 2)

    # Inputs Atitude Yaw
    criar_input(frame_yaw, "Kp:", "pid_yaw_att_kp", 2.0, 0)
    criar_input(frame_yaw, "Ki:", "pid_yaw_att_ki", 2.0, 1)
    criar_input(frame_yaw, "Kd:", "pid_yaw_att_kd", 0.5, 2)

    # ================= ABA 2: CENÁRIO =================
    frame_init = tk.LabelFrame(tab_voo, text="Condições Iniciais", font=("Arial", 9, "bold"), padx=10, pady=5)
    frame_init.grid(row=0, column=0, padx=10, pady=10, sticky="n")

    frame_set = tk.LabelFrame(tab_voo, text="Setpoint (Alvo)", font=("Arial", 9, "bold"), padx=10, pady=5)
    frame_set.grid(row=0, column=1, padx=10, pady=10, sticky="n")

    frame_vento = tk.LabelFrame(tab_voo, text="Parâmetros do Vento", font=("Arial", 9, "bold"), padx=10, pady=5)
    frame_vento.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    # Iniciais
    criar_input(frame_init, "X (m):", "init_x", 0.0, 0)
    criar_input(frame_init, "Y (m):", "init_y", 0.0, 1)
    criar_input(frame_init, "Z (m):", "init_z", 30.0, 2)
    criar_input(frame_init, "Roll (°):", "init_roll", 0.0, 3)
    criar_input(frame_init, "Pitch (°):", "init_pitch", 0.0, 4)
    criar_input(frame_init, "Yaw (°):", "init_yaw", 0.0, 5)

    # Setpoints
    criar_input(frame_set, "X (m):", "setpoint_x", 0.0, 0)
    criar_input(frame_set, "Y (m):", "setpoint_y", 0.0, 1)
    criar_input(frame_set, "Z (m):", "setpoint_z", 30.0, 2)
    criar_input(frame_set, "Yaw (°):", "setpoint_yaw", 0.0, 3)

    # Vento (Layout manual para alinhar melhor)
    criar_input(frame_vento, "Vento X (m/s):", "vento_x", 0.0, 0)
    criar_input(frame_vento, "Vento Y (m/s):", "vento_y", 0.0, 1)
    criar_input(frame_vento, "Vento Z (m/s):", "vento_z", 0.0, 2)
    
    # Inputs de tempo do vento na coluna da direita do frame de vento
    tk.Label(frame_vento, text="Início (s):").grid(row=0, column=2, padx=5, sticky="e")
    var_t0 = tk.StringVar(value="10.0"); tk.Entry(frame_vento, textvariable=var_t0, width=8).grid(row=0, column=3, padx=5); vars_dict["t0_vento"] = var_t0
    
    tk.Label(frame_vento, text="Duração (s):").grid(row=1, column=2, padx=5, sticky="e")
    var_dur = tk.StringVar(value="5.0"); tk.Entry(frame_vento, textvariable=var_dur, width=8).grid(row=1, column=3, padx=5); vars_dict["duração_vento"] = var_dur

    # ================= ABA 3: FÍSICA E INICIAL =================
    frame_fisica = tk.LabelFrame(parametros, text="Modelo do Drone (Mecânico)", font=("Arial", 9, "bold"), padx=10, pady=5)
    frame_fisica.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

    frame_motor = tk.LabelFrame(parametros, text="Motores e Hélices", font=("Arial", 9, "bold"), padx=10, pady=5)
    frame_motor.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

    frame_aero = tk.LabelFrame(parametros, text="Aerodinâmica (Arrasto)", font=("Arial", 9, "bold"), padx=10, pady=5)
    frame_aero.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    frame_sim = tk.LabelFrame(parametros, text="Configuração da Simulação", font=("Arial", 9, "bold"), padx=10, pady=5)
    frame_sim.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

    # Drone Mecânico
    criar_input(frame_fisica, "Massa (kg):", "massa", 1.0, 0)
    criar_input(frame_fisica, "Braço L (m):", "braco_l", 0.25, 1)
    criar_input(frame_fisica, "Ixx (kg.m²):", "Ixx", 0.015, 2)
    criar_input(frame_fisica, "Iyy (kg.m²):", "Iyy", 0.015, 3)
    criar_input(frame_fisica, "Izz (kg.m²):", "Izz", 0.020, 4)

    # Motores
    criar_input(frame_motor, "Máximo RPM:", "RPM_max", 8500, 0)
    criar_input(frame_motor, "Coef. Empuxo (Kf):", "K_f", 2.0e-5, 1)
    criar_input(frame_motor, "Coef. Arrasto (Km):", "K_m", 2.0e-6, 2)

    # Aerodinâmica
    criar_input(frame_aero, "Coef. Arrasto Cx:", "C_x", 0.03, 0)
    criar_input(frame_aero, "Coef. Arrasto Cy:", "C_y", 0.03, 1)
    criar_input(frame_aero, "Coef. Arrasto Cz:", "C_z", 0.1, 2)

    # Simulação
    criar_input(frame_sim, "Tempo Total (s):", "sim_time", 20.0, 0)
    criar_input(frame_sim, "Passo de Integração (s):", "dt", 0.001, 1)

    # ================= BOTÃO INICIAR (RODAPÉ) =================
    frame_bottom = tk.Frame(root)
    frame_bottom.pack(side="bottom", fill="x", pady=10)

    def on_start():
        try:
            # Converter tudo para float e salvar no dicionário final
            for key, var in vars_dict.items():
                config_final[key] = float(var.get())

            root.destroy() # Fecha a janela e libera o código principal
        except ValueError:
            tk.messagebox.showerror("Erro de Formato", "Por favor, certifique-se de que todos os campos contêm apenas números (use ponto '.' para decimais).")

    btn_start = tk.Button(frame_bottom, text="INICIAR SIMULAÇÃO 🚀", command=on_start, 
                          bg="#007acc", fg="white", font=("Segoe UI", 12, "bold"), 
                          relief="flat", cursor="hand2")
    btn_start.pack(ipadx=30, ipady=10)

    root.mainloop()
    return config_final

# ============================================================================
# CLASSE: CONTROLADOR PID
# ============================================================================
class PIDController:
    """
    Controlador PID genérico para controle de uma variável.
    """
    def __init__(self, Kp, Ki, Kd, setpoint, dt):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint, self.dt = setpoint, dt
        self.termo_integral = 0.0
        self.erro_anterior = 0.0

    def update(self, valor_medido, circular=False):
        """Calcula a saída do controlador PID."""
        erro = self.setpoint - valor_medido

        if circular:
            erro = (erro + np.pi) % (2 * np.pi) - np.pi 

        self.termo_integral += (erro + self.erro_anterior) * (self.dt / 2.0)
        derivativo = (erro - self.erro_anterior) / self.dt
        self.erro_anterior = erro
        return (self.Kp * erro) + (self.Ki * self.termo_integral) + (self.Kd * derivativo)

# ============================================================================
# CLASSE: MODELO FÍSICO DO DRONE
# ============================================================================
class QuadcopterDynamics:
    """
    Modelo dinâmico completo de um quadricóptero com arrasto aerodinâmico.
    """
    def __init__(self, dt, config):
        # Parâmetros de simulação
        self.dt = dt
        
        # Parâmetros físicos básicos
        self.m = config.get('massa')    # Massa (kg)
        self.g = 9.81   # Gravidade (m/s²)
        self.L = config.get('braco_l')  # Distância do centro aos motores (m)

        # Momentos de inércia
        self.Ixx = config.get('Ixx')
        self.Iyy = config.get('Iyy')
        self.Izz = config.get('Izz')

        # Condições Iniciais (Vindas da Interface)
        x0 = config.get('init_x')
        y0 = config.get('init_y')
        z0 = config.get('init_z')
        roll0 = np.deg2rad(config.get('init_roll'))
        pitch0 = np.deg2rad(config.get('init_pitch'))
        yaw0 = np.deg2rad(config.get('init_yaw'))
        
        # Estados do drone
        self.pos = np.array([x0, y0, z0])      # Posição [x, y, z]
        self.vel = np.array([0.0, 0.0, 0.0])       # Velocidade linear
        self.angulos = np.array([roll0, pitch0, yaw0])  # [roll, pitch, yaw]
        self.ang_vel = np.array([0.0, 0.0, 0.0])   # Velocidade angular

        # Coeficientes dos rotores
        self.k_f = config.get('K_f')  # Coeficiente de empuxo
        self.k_m = config.get('K_m') # Coeficiente de momento
        
        # Saturação dos motores
        self.MAX_RPM = config.get('RPM_max')
        self.MAX_OMEGA = self.MAX_RPM * 2 * np.pi / 60
        
        # Modelo de arrasto aerodinâmico
        self.C_drag = np.array([config.get('C_x'), config.get('C_y'), config.get('C_z')])  # Coeficientes de arrasto [x, y, z]
        self.drag_force = np.array ([0.0, 0.0, 0.0])
        self.rho = 1.225                            # Densidade do ar (kg/m³)

    def update(self, thrust, torques, velocidade_vento):
        """
        Atualiza o estado físico do drone dado empuxo total, torques e vento.
        """
        # ========== DINÂMICA ROTACIONAL ==========
        T = thrust
        tau_phi, tau_theta, tau_psi = torques
        phi, theta, psi = self.angulos
        p, q, r = self.ang_vel
        
        # Equações de Euler para corpo rígido
        p_dot = ((self.Iyy - self.Izz) / self.Ixx) * q * r + tau_phi / self.Ixx
        q_dot = ((self.Izz - self.Ixx) / self.Iyy) * p * r + tau_theta / self.Iyy
        r_dot = ((self.Ixx - self.Iyy) / self.Izz) * p * q + tau_psi / self.Izz
        
        self.ang_vel += np.array([p_dot, q_dot, r_dot]) * self.dt
        self.angulos += self.ang_vel * self.dt

        # Normalização do ângulo de yaw entre -180 a 180 graus
        self.angulos[2] = np.arctan2(np.sin(self.angulos[2]), np.cos(self.angulos[2]))
        
        # ========== MATRIZ DE ROTAÇÃO ==========
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        
        # Matriz R: Corpo -> Inercial (ZYX Euler)
        R = np.array([
            [c_psi*c_theta, c_psi*s_theta*s_phi - s_psi*c_phi, c_psi*s_theta*c_phi + s_psi*s_phi],
            [s_psi*c_theta, s_psi*s_theta*s_phi + c_psi*c_phi, s_psi*s_theta*c_phi - c_psi*s_phi],
            [-s_theta,      c_theta*s_phi,                     c_theta*c_phi]
        ])
        
        # ========== FORÇAS NO REFERENCIAL INERCIAL ==========
        force_inertial = R @ np.array([0, 0, T])
        gravity_force = np.array([0, 0, -self.m * self.g])
        
        # ========== ARRASTO AERODINÂMICO ==========
        v_ar_inercial = self.vel - velocidade_vento
        v_corpo = R.T @ v_ar_inercial
        drag_force_corpo = np.array([0.0, 0.0, 0.0])
        
        # Modelo puramente quadrático de arrasto
        velocidade_ao_quadrado = v_corpo * np.abs(v_corpo)
        drag_force_corpo = -0.5 * self.rho * self.C_drag * velocidade_ao_quadrado
            
        # Transformação para referencial inercial
        drag_force = R @ drag_force_corpo
        self.drag_force = drag_force

        # ========== DINÂMICA TRANSLACIONAL ==========
        total_force = force_inertial + gravity_force + drag_force
        acceleration = total_force / self.m
        
        self.vel += acceleration * self.dt
        self.pos += self.vel * self.dt
        
        # Condição de contorno: solo
        if self.pos[2] < 0:
            self.pos[2] = 0
            self.vel[2] = 0


# ============================================================================
# FUNÇÕES AUXILIARES PARA VISUALIZAÇÃO E CONVERSÃO DE ÂNGULOS
# ============================================================================

def menor_angulo(erro):
    # Força o erro para o intervalo [-180, 180]
    return (erro + 180) % 360 - 180

def plotar_resumo_parametros(config):
    """
    Gera uma figura limpa listando todos os parâmetros utilizados na simulação.
    """
    fig, ax = plt.subplots(figsize=(11, 7)) # Tamanho da tela
    fig.canvas.manager.set_window_title('Resumo da Configuração')
    ax.axis('off') 

    # Título Principal
    ax.text(0.5, 0.95, "CONFIGURAÇÃO DA SIMULAÇÃO", 
            ha='center', va='top', fontsize=14, weight='bold', transform=ax.transAxes)

    # --- COLUNA 1: CONTROLADORES PID ---
    x_col1 = 0.02
    y_start = 0.85
    spacing = 0.045
    
    ax.text(x_col1, y_start, "1. SINTONIA PID", fontsize=11, weight='bold', color='#003366', transform=ax.transAxes)
    
    # Texto formatado
    text_pid = (
        f"Posição (X, Y):\n"
        f"  Kp: {config['pid_pos_kp']} | Ki: {config['pid_pos_ki']} | Kd: {config['pid_pos_kd']}\n\n"
        f"Altitude (Z):\n"
        f"  Kp: {config['pid_z_kp']} | Ki: {config['pid_z_ki']} | Kd: {config['pid_z_kd']}\n\n"
        f"Atitude (Roll/Pitch):\n"
        f"  Kp: {config['pid_att_kp']} | Ki: {config['pid_att_ki']} | Kd: {config['pid_att_kd']}\n\n"
        f"Atitude (Yaw):\n"
        f"  Kp: {config['pid_yaw_att_kp']} | Ki: {config['pid_yaw_att_ki']} | Kd: {config['pid_yaw_att_kd']}"
    )
    ax.text(x_col1, y_start - spacing, text_pid, va='top', fontsize=10, family='monospace', transform=ax.transAxes)

    # --- COLUNA 2: CENÁRIO E VENTO ---
    x_col2 = 0.33
    ax.text(x_col2, y_start, "2. CENÁRIO DE TESTE", fontsize=11, weight='bold', color='#003366', transform=ax.transAxes)
    
    text_cenario = (
        f"Setpoints (Alvo):\n"
        f"  X: {config['setpoint_x']} m\n"
        f"  Y: {config['setpoint_y']} m\n"
        f"  Z: {config['setpoint_z']} m\n"
        f"  Yaw: {config['setpoint_yaw']}°\n\n"
        f"Condições Iniciais:\n"
        f"  Pos: [{config['init_x']}, {config['init_y']}, {config['init_z']}] m\n"
        f"  Ang: [{config['init_roll']}°, {config['init_pitch']}°, {config['init_yaw']}°]\n\n"
        f"Simulação:\n"
        f"  Tempo Total: {config['sim_time']} s\n"
        f"  Passo (dt): {config['dt']} s"
    )
    ax.text(x_col2, y_start - spacing, text_cenario, va='top', fontsize=10, family='monospace', transform=ax.transAxes)

    # Destaque para o Vento
    y_vento = 0.45
    ax.text(x_col2, y_vento, "3. PERTURBAÇÃO (VENTO)", fontsize=11, weight='bold', color='#003366', transform=ax.transAxes)
    text_vento = (
        f"Velocidade:\n"
        f"  Vx: {config['vento_x']} m/s\n"
        f"  Vy: {config['vento_y']} m/s\n"
        f"  Vz: {config['vento_z']} m/s\n\n"
        f"Atuação:\n"
        f"  Início: {config['t0_vento']} s\n"
        f"  Duração: {config['duração_vento']} s"
    )
    ax.text(x_col2, y_vento - spacing, text_vento, va='top', fontsize=10, family='monospace', transform=ax.transAxes)

    # --- COLUNA 3: FÍSICA E DRONE ---
    x_col3 = 0.64
    ax.text(x_col3, y_start, "4. MODELO FÍSICO", fontsize=11, weight='bold', color='#003366', transform=ax.transAxes)
    
    text_fisica = (
        f"Mecânica:\n"
        f"  Massa: {config['massa']} kg\n"
        f"  Braço (L): {config['braco_l']} m\n"
        f"  Ixx: {config['Ixx']} kg.m²\n"
        f"  Iyy: {config['Iyy']} kg.m²\n"
        f"  Izz: {config['Izz']} kg.m²\n\n"
        f"Propulsão:\n"
        f"  Max RPM: {config['RPM_max']}\n"
        f"  Kf: {config['K_f']:.1e}\n"
        f"  Km: {config['K_m']:.1e}\n\n"
        f"Aerodinâmica:\n"
        f"  Cx, Cy: {config['C_x']}, {config['C_y']}\n"
        f"  Cz: {config['C_z']}"
    )
    ax.text(x_col3, y_start - spacing, text_fisica, va='top', fontsize=10, family='monospace', transform=ax.transAxes)

    # Rodapé
    ax.text(0.5, 0.02, "SIMULAÇÃO DE CONTROLE E PERFORMANCE DE UM DRONE QUADRICÓPTERO - TCC - HENRIQUE SILVA", 
            ha='center', fontsize=8, color='gray', transform=ax.transAxes)

    plt.tight_layout()

def criar_matriz_rotacao(roll, pitch, yaw):
    """
    Cria matriz de rotação 3D dados os ângulos de Euler.
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def plotar_dashboard_principal(time_axis, historico_pos, historico_setpoints, historico_ang, historico_comando_angulo):
    """
    Cria dashboard com posição e atitude do drone.
    """
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.canvas.manager.set_window_title('Dashboard Principal')
    fig.suptitle('Dashboard Principal com Controle de Posição', fontsize=16)
    
    # Posição X
    axs[0].plot(time_axis, historico_pos[:, 0], label='Posição X Real', color='blue')
    axs[0].plot(time_axis, historico_setpoints[:, 0], color='red', linestyle='--', label='Setpoint X')
    axs[0].set_ylabel('Posição X [m]')
    axs[0].grid(True)
    axs[0].legend()
    
    # Posição Y
    axs[1].plot(time_axis, historico_pos[:, 1], label='Posição Y Real', color='green')
    axs[1].plot(time_axis, historico_setpoints[:, 1], color='red', linestyle='--', label='Setpoint Y')
    axs[1].set_ylabel('Posição Y [m]')
    axs[1].grid(True)
    axs[1].legend()
    
    # Altitude Z
    axs[2].plot(time_axis, historico_pos[:, 2], label='Altitude (Z) Real', color='purple')
    axs[2].plot(time_axis, historico_setpoints[:, 2], color='red', linestyle='--', label='Setpoint Z')
    axs[2].set_ylabel('Altitude [m]')
    axs[2].grid(True)
    axs[2].legend()
    
    # Ângulos
    axs[3].plot(time_axis, historico_ang[:, 0], label='Roll', color='orange')
    axs[3].plot(time_axis, historico_ang[:, 1], label='Pitch', color='brown')
    axs[3].plot(time_axis, historico_ang[:, 2], label='Yaw', color='green')
    axs[3].plot(time_axis, historico_comando_angulo[:, 0], '--', label='Roll Ref', color='orange', alpha=0.7)
    axs[3].plot(time_axis, historico_comando_angulo[:, 1], '--', label='Pitch Ref', color='brown', alpha=0.7)
    axs[3].plot(time_axis, historico_comando_angulo[:, 2], '--', label='Yaw Ref', color='green', alpha=0.7)
    axs[3].set_xlabel('Tempo [s]')
    axs[3].set_ylabel('Ângulos [graus]')
    axs[3].grid(True)
    axs[3].legend()
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def plotar_dashboard_rotores(time_axis, historico_empuxo_rotor, historico_velocidade_rotor):
    """
    Cria dashboard com empuxo e RPM de cada rotor.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    fig.canvas.manager.set_window_title('Dashboard dos Rotores')
    fig.suptitle('Dashboard de Empuxo e RPM de Cada Rotor', fontsize=16)

    colors = ['C0', 'C1', 'C2', 'C3']
    titles = ['Rotor 1 (FE-CW)', 'Rotor 2 (FD-CCW)', 'Rotor 3 (TE-CCW)', 'Rotor 4 (TD-CW)']
    
    for i, ax in enumerate(axs.flat):
        ax.set_ylabel('Empuxo [N]', color=colors[i])
        ax.plot(time_axis, historico_empuxo_rotor[:, i], color=colors[i])
        ax.tick_params(axis='y', labelcolor=colors[i])
        ax.grid(True)
        ax.set_title(titles[i])

        ax_twin = ax.twinx()
        ax_twin.set_ylabel('RPM', color='gray')
        ax_twin.plot(time_axis, historico_velocidade_rotor[:, i], color='gray', linestyle='--')
        ax_twin.tick_params(axis='y', labelcolor='gray')

        ax.set_xlabel('Tempo [s]')
        ax.tick_params(labelbottom=True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])


def criar_animacao_3d(drone, historico_pos, historico_setpoints, historico_ang, historico_vento, historico_arrasto, steps):
    """
    Cria animação 3D da trajetória do drone.
    """
    print("Preparando animação...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Configuração dos limites
    ax.set_xlim(np.min(historico_pos[:, 0]) - 2, np.max(historico_pos[:, 0]) + 2)
    ax.set_ylim(np.min(historico_pos[:, 1]) - 2, np.max(historico_pos[:, 1]) + 2)
    ax.set_zlim(0, np.max(historico_pos[:, 2]) + 5)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Animação da Trajetória do Drone')
    ax.view_init(elev=20, azim=45)

    # Trajetórias
    trajetoria_desejada, = ax.plot(historico_setpoints[:, 0], historico_setpoints[:, 1], 
                                   historico_setpoints[:, 2], 'r--', linewidth=1, label='Trajetória Desejada')
    trajetoria_real, = ax.plot([], [], [], 'b-', linewidth=2, label='Trajetória Real')

    # Estrutura do drone (braços em X)
    tamanho_braco = drone.L
    d = tamanho_braco / np.sqrt(2)
    bracos_corpo = np.array([
        [-d, -d, 0],
        [ d, d, 0],
        [-d, d, 0],
        [d, -d, 0]
    ])
    linha_x, = ax.plot([], [], [], 'k-', linewidth=3)
    linha_y, = ax.plot([], [], [], 'k-', linewidth=3)
    
    # Indicador de direção frontal
    frente_corpo = np.array([[0, 0, 0], [0.4, 0, 0]])
    seta_frente, = ax.plot([],[],[], 'r-', linewidth=2)
    
    # Texto de informação de vento
    texto_vento = ax.text2D(0.05, 0.95, "Vento: 0.0 m/s", transform=ax.transAxes, color='blue', fontsize=12)
    texto_arrasto = ax.text2D(0.05, 0.92, "Arrasto: 0.0 N", transform=ax.transAxes, color='blue', fontsize=12)

    # Grid para campo de vetores do vento
    grid_size = 3
    x_grid = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], grid_size)
    y_grid = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], grid_size)
    z_grid = np.linspace(5, ax.get_zlim()[1] - 5, grid_size)
    X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid)
    
    campo_vento = None
    ax.legend()
    velocidade_animacao = 100

    def animate(i):
        nonlocal campo_vento
        frame_index = i * velocidade_animacao
        if frame_index >= steps:
            frame_index = steps - 1

        # Atualizar trajetória
        trajetoria_real.set_data(historico_pos[:frame_index, 0], historico_pos[:frame_index, 1])
        trajetoria_real.set_3d_properties(historico_pos[:frame_index, 2])

        # Posição e orientação atual
        pos = historico_pos[frame_index]
        roll, pitch, yaw = np.deg2rad(historico_ang[frame_index])

        # Rotacionar estrutura do drone
        R = criar_matriz_rotacao(roll, pitch, yaw)
        bracos_mundo = (R @ bracos_corpo.T).T + pos

        linha_x.set_data([bracos_mundo[0,0], bracos_mundo[1,0]], [bracos_mundo[0,1], bracos_mundo[1,1]])
        linha_x.set_3d_properties([bracos_mundo[0,2], bracos_mundo[1,2]])

        linha_y.set_data([bracos_mundo[2,0], bracos_mundo[3,0]], [bracos_mundo[2,1], bracos_mundo[3,1]])
        linha_y.set_3d_properties([bracos_mundo[2,2], bracos_mundo[3,2]])

        # Seta indicando frente
        frente_mundo = (R @ frente_corpo.T).T + pos
        seta_frente.set_data([frente_mundo[0,0], frente_mundo[1,0]],
                            [frente_mundo[0,1], frente_mundo[1,1]])
        seta_frente.set_3d_properties([frente_mundo[0,2], frente_mundo[1,2]])

        # Atualizar campo de vento
        vento_vector = historico_vento[frame_index]
        arrasto_vector = historico_arrasto[frame_index]
        mag_vento = np.linalg.norm(vento_vector)
        mag_arrasto = np.linalg.norm(arrasto_vector)
        texto_vento.set_text(f"Vento: {mag_vento:.1f} m/s")
        texto_arrasto.set_text(f"Arrasto: {mag_arrasto:.1f} N")

        if campo_vento is not None:
            campo_vento.remove()
            campo_vento = None
        
        if mag_vento > 0.1:
            U_grid = np.full_like(X_grid, vento_vector[0])
            V_grid = np.full_like(Y_grid, vento_vector[1])
            W_grid = np.full_like(Z_grid, vento_vector[2])
            
            campo_vento = ax.quiver(X_grid, Y_grid, Z_grid, 
                                    U_grid, V_grid, W_grid,
                                    length=0.3, normalize=False, color='blue', 
                                    alpha=0.3, arrow_length_ratio=0.2)

        return trajetoria_real, linha_x, linha_y, seta_frente, texto_vento, texto_arrasto

    num_frames = steps // velocidade_animacao
    ani = FuncAnimation(fig, animate, frames=num_frames, blit=False, interval=50)
    return ani


# ============================================================================
# FUNÇÃO PRINCIPAL DE SIMULAÇÃO
# ============================================================================
def main():
    """
    Função principal que executa a simulação completa.
    """
    # ========== INICIALIZAÇÃO ==========
    config = abrir_configuracao()

    if not config:
        print("Simulação cancelada pelo usuário.")
        return

    drone = QuadcopterDynamics(dt=config['dt'], config=config)
    sim_time = config['sim_time']
    steps = int(sim_time / drone.dt)
    
    x_inicial = drone.pos[0]
    y_inicial = drone.pos[1]
    z_inicial = drone.pos[2]
    
    # ========== CONTROLADORES PID ==========
    # PIDs de Posição (malha externa)
    pid_x = PIDController(config['pid_pos_kp'], config['pid_pos_ki'], config['pid_pos_kd'], 0, drone.dt)
    pid_y = PIDController(config['pid_pos_kp'], config['pid_pos_ki'], config['pid_pos_kd'], 0, drone.dt)
    pid_z = PIDController(config['pid_z_kp'],   config['pid_z_ki'],   config['pid_z_kd'],   0, drone.dt)
    
    # PIDs de Atitude (malha interna)
    pid_roll  = PIDController(config['pid_att_kp'], config['pid_att_ki'], config['pid_att_kd'], 0.0, drone.dt)
    pid_pitch = PIDController(config['pid_att_kp'], config['pid_att_ki'], config['pid_att_kd'], 0.0, drone.dt)
    pid_yaw = PIDController(config['pid_yaw_att_kp'], config['pid_yaw_att_ki'], config['pid_yaw_att_kd'], 0.0, drone.dt)

    # ========== ARRAYS DE HISTÓRICO ==========
    historico_pos = np.zeros((steps, 3))
    historico_ang = np.zeros((steps, 3))
    historico_empuxo_rotor = np.zeros((steps, 4))
    historico_velocidade_rotor = np.zeros((steps, 4))
    historico_comando_angulo = np.zeros((steps, 3))
    historico_setpoints = np.zeros((steps, 3))
    historico_vento = np.zeros((steps, 3))
    historico_arrasto = np.zeros((steps, 3))
    time_axis = np.linspace(0, sim_time, steps)
    
    # ========== PARÂMETROS DE VENTO ==========
    TEMPO_INICIO_VENTO = config['t0_vento']
    VELOCIDADE_VENTO = np.array([config['vento_x'], config['vento_y'], config['vento_z']])  # [m/s]
    TEMPO_DURACAO_VENTO = config['duração_vento']

    # ========== LOOP PRINCIPAL DE SIMULAÇÃO ==========
    for i in range(steps):
        current_time = time_axis[i]
        
        # --- Definir Setpoints ---

        if current_time < 2.0:
            setpoint_x = config['init_x']
            setpoint_y = config['init_y']
            setpoint_z = config['init_z']
            setpoint_yaw = np.deg2rad(config['init_yaw'])
        else:
            setpoint_x = config['setpoint_x']
            setpoint_y = config['setpoint_y']
            setpoint_z = config['setpoint_z']
            setpoint_yaw = np.deg2rad(config['setpoint_yaw'])

        historico_setpoints[i, :] = [setpoint_x, setpoint_y, setpoint_z]
        
        # --- Calcular Vento Atual ---
        vento_atual = np.array([0.0, 0.0, 0.0])
        if TEMPO_INICIO_VENTO <= current_time < TEMPO_INICIO_VENTO + TEMPO_DURACAO_VENTO:
            vento_atual = VELOCIDADE_VENTO
        
        historico_vento[i, :] = vento_atual
        
        # --- Atualizar Setpoints dos PIDs ---
        pid_x.setpoint = setpoint_x
        pid_y.setpoint = setpoint_y
        pid_z.setpoint = setpoint_z
        pid_yaw.setpoint = setpoint_yaw
        
        pos, angles = drone.pos, drone.angulos
        
# --- Controle em Cascata: Posição -> Atitude ---
        
        # PIDs de Posição calculam a "Força Desejada" no referencial Global
        fx_inercial = pid_x.update(pos[0]) 
        fy_inercial = pid_y.update(pos[1])
        
        # Rotação do vetor de força para o referencial do Corpo (Drone)
        yaw = angles[2]
        c_yaw = np.cos(yaw)
        s_yaw = np.sin(yaw)
        
        fx_body = fx_inercial * c_yaw + fy_inercial * s_yaw
        fy_body = -fx_inercial * s_yaw + fy_inercial * c_yaw
        
        # Conversão da Força do Corpo para Ângulos de Comando
        pitch_cmd = fx_body
        roll_cmd = -fy_body
        
        # Limitar comandos de ângulo (Segurança)
        roll_cmd = np.clip(roll_cmd, -np.deg2rad(45), np.deg2rad(45))
        pitch_cmd = np.clip(pitch_cmd, -np.deg2rad(45), np.deg2rad(45))

        pid_roll.setpoint = roll_cmd
        pid_pitch.setpoint = pitch_cmd
        
        # --- Controle de Atitude ---
        tau_phi = pid_roll.update(angles[0])
        tau_theta = pid_pitch.update(angles[1])
        tau_psi = pid_yaw.update(angles[2], circular=True)
        
        # --- Controle de Altitude ---
        thrust_correction = pid_z.update(pos[2])
        total_thrust = max(0, drone.m * drone.g * thrust_correction)
        t_base = total_thrust / 4.0
        
        # --- Alocação de Controle (Mixing) ---
        ratio_torque_thrust = drone.k_m / drone.k_f
        
        tau_roll  = tau_phi   / (4 * drone.L)
        tau_pitch = tau_theta / (4 * drone.L)
        tau_yaw   = tau_psi   / (4 * ratio_torque_thrust)

        # Distribuição nas 4 rotores (Configuração Quad-X)
        f_ref_1 = max(0, t_base - tau_pitch + tau_roll + tau_yaw)  # Frente-Esquerda (CW)
        f_ref_2 = max(0, t_base - tau_pitch - tau_roll - tau_yaw)  # Frente-Direita (CCW)
        f_ref_3 = max(0, t_base + tau_pitch + tau_roll - tau_yaw)  # Trás-Esquerda (CCW)
        f_ref_4 = max(0, t_base + tau_pitch - tau_roll + tau_yaw)  # Trás-Direita (CW)

        # --- Conversão para Velocidades Angulares dos Motores ---
        omega_1 = np.clip(np.sqrt(f_ref_1 / drone.k_f), 0, drone.MAX_OMEGA)
        omega_2 = np.clip(np.sqrt(f_ref_2 / drone.k_f), 0, drone.MAX_OMEGA)
        omega_3 = np.clip(np.sqrt(f_ref_3 / drone.k_f), 0, drone.MAX_OMEGA)
        omega_4 = np.clip(np.sqrt(f_ref_4 / drone.k_f), 0, drone.MAX_OMEGA)

        rpm_1 = (omega_1 * 60) / (2 * np.pi)
        rpm_2 = (omega_2 * 60) / (2 * np.pi)
        rpm_3 = (omega_3 * 60) / (2 * np.pi)
        rpm_4 = (omega_4 * 60) / (2 * np.pi)

        # --- Cálculo de Forças e Torques Reais ---
        f_real_1 = drone.k_f * (omega_1**2)
        f_real_2 = drone.k_f * (omega_2**2)
        f_real_3 = drone.k_f * (omega_3**2)
        f_real_4 = drone.k_f * (omega_4**2)

        empuxo_total = f_real_1 + f_real_2 + f_real_3 + f_real_4

        m_1 = drone.k_m * (omega_1**2)
        m_2 = drone.k_m * (omega_2**2)
        m_3 = drone.k_m * (omega_3**2)
        m_4 = drone.k_m * (omega_4**2)

        tau_phi_real = drone.L * ((f_real_1 + f_real_3) - (f_real_2 + f_real_4))
        tau_theta_real = drone.L * ((f_real_3 + f_real_4) - (f_real_1 + f_real_2))
        tau_psi_real = (m_1 + m_4) - (m_2 + m_3)

        # --- Registrar Histórico ---
        historico_empuxo_rotor[i, :] = [f_real_1, f_real_2, f_real_3, f_real_4]
        historico_velocidade_rotor[i, :] = [rpm_1, rpm_2, rpm_3, rpm_4]
        historico_comando_angulo[i, :] = [np.rad2deg(roll_cmd), np.rad2deg(pitch_cmd), pid_yaw.setpoint]

        # --- Atualizar Dinâmica do Drone ---
        real_torques = np.array([tau_phi_real, tau_theta_real, tau_psi_real])
        drone.update(empuxo_total, real_torques, velocidade_vento=vento_atual)
        
        historico_pos[i, :] = drone.pos
        historico_ang[i, :] = np.rad2deg(drone.angulos)
        historico_arrasto[i, :] = drone.drag_force

    # ========== VISUALIZAÇÃO ==========
    print("\nGerando visualizações...")

    #Tela dos Inputs
    plotar_resumo_parametros(config)
    
    # Dashboard principal
    plotar_dashboard_principal(time_axis, historico_pos, historico_setpoints, historico_ang, historico_comando_angulo)
    
    # Dashboard dos rotores
    plotar_dashboard_rotores(time_axis, historico_empuxo_rotor, historico_velocidade_rotor)
    
    # Animação 3D
    ani = criar_animacao_3d(drone, historico_pos, historico_setpoints, historico_ang, historico_vento, historico_arrasto, steps)
    
    plt.show()


# ============================================================================
# PONTO DE ENTRADA
# ============================================================================
if __name__ == "__main__":
    main()
