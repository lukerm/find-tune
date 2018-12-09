## Edited from: https://github.com/devicehive/devicehive-audio-analysis/blob/master/log_config.py

LOGGING = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '[%(levelname)s] %(asctime)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
    },
    'loggers': {
        'audio_analysis': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False
        },
        'dh_webconfig': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False
        },
        'devicehive': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False
        },
    }
}