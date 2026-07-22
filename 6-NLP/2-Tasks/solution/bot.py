"""
Marvin the Bot - Análisis de Sentimiento con TextBlob
=====================================================
Un chatbot que analiza el sentimiento del texto del usuario y responde
accordingmente. Usa TextBlob para detectar polaridad (positivo/negativo)
y extraer frases sustantivas para hacer preguntas de seguimiento.

Polaridad: -1 (muy negativo) a +1 (muy positivo)
  -1 -------- -0.5 -------- 0 -------- 0.5 -------- 1
    muy mal     mal      neutral     bien      muy bien
"""

# =============================================================================
# IMPORTACIONES
# =============================================================================
import random  # No se usa en este bot, pero está disponible para variantes
from textblob import TextBlob  # Librería de NLP: sentimiento, traducción, etc.
from textblob.np_extractors import ConllExtractor  # Extractor preciso de noun phrases

# Crear instancia del extractor (usa el corpus CoNLL para mejor precisión)
extractor = ConllExtractor()

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
def main():   
    # -------------------------------------------------------------------------
    # PASO 1: Presentación del bot
    # -------------------------------------------------------------------------
    print("Hello, I am Marvin, the friendly robot.")
    print("You can end this conversation at any time by typing 'bye'")    
    print("After typing each answer, press 'enter'")
    print("How are you today?")

    # -------------------------------------------------------------------------
    # PASO 2: Loop principal de conversación
    # -------------------------------------------------------------------------
    while True:  # Loop infinito hasta que el usuario escriba "bye"
        
        # Esperar input del usuario
        # En terminal: muestra ">" y espera
        # En Jupyter: muestra una caja de texto arriba del notebook
        user_input = input("> ")

        # -------------------------------------------------------------------------
        # PASO 3: Condición de salida
        # -------------------------------------------------------------------------
        # .lower() convierte "BYE", "ByE", "bye" → "bye"
        # Si coincide, break rompe el while True
        if user_input.lower() == "bye":            
            break
        
        # -------------------------------------------------------------------------
        # PASO 4: Análisis de sentimiento y extracción de entidades
        # -------------------------------------------------------------------------
        else:
            # Crear objeto TextBlob con el texto del usuario
            # np_extractor=extractor usa el modelo CoNLL para extraer noun phrases
            user_input_blob = TextBlob(user_input, np_extractor=extractor)                        
            
            # Extraer frases sustantivas (noun phrases)
            # Ejemplos:
            #   "my cat is happy" → ["cat"]
            #   "I love programming" → ["programming"]
            #   "hello" → [] (vacío)
            np = user_input_blob.noun_phrases
            
            # -------------------------------------------------------------------------
            # PASO 5: Generar respuesta según sentimiento
            # -------------------------------------------------------------------------
            response = ""
            
            # polarity va de -1 a 1
            # TextBlob calcula esto usando un diccionario de palabras con pesos
            if user_input_blob.polarity <= -0.5:
                # Muy negativo: "I hate this", "This is terrible"
                response = "Oh dear, that sounds bad. "
            elif user_input_blob.polarity <= 0:
                # Ligeramente negativo: "It's okay I guess", "Not great"
                response = "Hmm, that's not great. "
            elif user_input_blob.polarity <= 0.5:
                # Ligeramente positivo: "It's fine", "I like it"
                response = "Well, that sounds positive. "
            elif user_input_blob.polarity <= 1:
                # Muy positivo: "I love this!", "This is amazing!"
                response = "Wow, that sounds great. "

            # -------------------------------------------------------------------------
            # PASO 6: Pregunta de seguimiento inteligente
            # -------------------------------------------------------------------------
            if len(np) != 0:
                # Si detectó noun phrases, pregunta sobre la primera
                # .pluralize() convierte: "cat" → "cats", "mouse" → "mice"
                response = response + "Can you tell me more about " + np[0].pluralize() + "?"
            else:
                # Si no detectó nada, pregunta genérica
                response = response + "Can you tell me more?"
            
            # Mostrar respuesta
            print(response)
    
    # -------------------------------------------------------------------------
    # PASO 7: Despedida
    # -------------------------------------------------------------------------
    print("It was nice talking to you, goodbye!")

# =============================================================================
# EJECUTAR
# =============================================================================
# main() inicia el programa
main()
