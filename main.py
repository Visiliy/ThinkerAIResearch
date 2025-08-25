import torch
import torch.nn as nn
from ConvBlock import ConvBlock, DynamicPooling, LinearPerformerAttention, LinearParameterizationKernel, FastKernelCompression, LinearBlockSparseAttention, OptimizedDilatedResidual, TokenMerging, LinearLocalAttention, LinearDynamicInceptionBlock


def test_conv_block():
    """Тестирование основного ConvBlock"""
    print("=== Тестирование ConvBlock ===")
    
    # Параметры теста
    batch_size = 2
    seq_len = 64
    dim = 256
    heads = 8
    
    # Создаем модель
    model = ConvBlock(dim=dim, heads=heads, dropout=0.1)
    
    # Создаем тестовые данные
    x = torch.randn(batch_size, seq_len, dim)
    
    print(f"Входные данные: {x.shape}")
    
    # Тестируем forward pass
    try:
        with torch.no_grad():
            output = model(x)
        print(f"✅ ConvBlock работает! Выходные данные: {output.shape}")
        print(f"   Вход: {x.shape} -> Выход: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Ошибка в ConvBlock: {e}")
        return False


def test_individual_components():
    """Тестирование отдельных компонентов"""
    print("\n=== Тестирование отдельных компонентов ===")
    
    batch_size = 2
    seq_len = 32
    dim = 128
    
    components = [
        ("DynamicPooling", DynamicPooling(output_size=16, mode='adaptive')),
        ("LinearPerformerAttention", LinearPerformerAttention(dim, heads=4, feature_dim=64)),
        ("LinearParameterizationKernel", LinearParameterizationKernel(dim, dim, feature_dim=32)),
        ("FastKernelCompression", FastKernelCompression(dim, reduction_ratio=4)),
        ("LinearBlockSparseAttention", LinearBlockSparseAttention(dim, block_size=16, heads=4, feature_dim=64)),
        ("OptimizedDilatedResidual", OptimizedDilatedResidual(dim, dilations=[1, 2, 4], dropout=0.1)),
        ("TokenMerging", TokenMerging(dim, reduction_ratio=2)),
        ("LinearLocalAttention", LinearLocalAttention(dim, window_size=5, heads=4, feature_dim=32)),
        ("LinearDynamicInceptionBlock", LinearDynamicInceptionBlock(dim, kernel_sizes=[1, 3], feature_dims=[32, 32]))
    ]
    
    success_count = 0
    
    for name, component in components:
        try:
            x = torch.randn(batch_size, seq_len, dim)
            with torch.no_grad():
                output = component(x)
            print(f"✅ {name}: {x.shape} -> {output.shape}")
            success_count += 1
        except Exception as e:
            print(f"❌ {name}: Ошибка - {e}")
    
    print(f"\nРезультат: {success_count}/{len(components)} компонентов работают корректно")
    return success_count == len(components)


def test_gradient_flow():
    """Тестирование потока градиентов"""
    print("\n=== Тестирование потока градиентов ===")
    
    batch_size = 1
    seq_len = 16
    dim = 64
    
    model = ConvBlock(dim=dim, heads=4, dropout=0.1)
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    
    try:
        output = model(x)
        loss = output.mean()
        loss.backward()
        
        print(f"✅ Градиенты успешно вычислены!")
        print(f"   Градиент входных данных: {x.grad.shape}")
        print(f"   Норма градиента: {x.grad.norm().item():.6f}")
        return True
    except Exception as e:
        print(f"❌ Ошибка при вычислении градиентов: {e}")
        return False


def test_memory_usage():
    """Тестирование использования памяти"""
    print("\n=== Тестирование использования памяти ===")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Тестирование на GPU")
    else:
        device = torch.device('cpu')
        print("Тестирование на CPU")
    
    batch_size = 4
    seq_len = 128
    dim = 512
    
    model = ConvBlock(dim=dim, heads=8, dropout=0.1).to(device)
    x = torch.randn(batch_size, seq_len, dim).to(device)
    
    try:
        # Очищаем кэш GPU если доступен
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            output = model(x)
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_used = final_memory - initial_memory
            print(f"✅ Память использована: {memory_used / 1024**2:.2f} MB")
        
        print(f"✅ Модель успешно работает на {device}")
        return True
    except Exception as e:
        print(f"❌ Ошибка при тестировании памяти: {e}")
        return False


def main():
    """Основная функция тестирования"""
    print("🧪 Начинаем тестирование ConvBlock")
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    
    # Запускаем все тесты
    tests = [
        test_conv_block,
        test_individual_components,
        test_gradient_flow,
        test_memory_usage
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Ошибка в тесте {test.__name__}: {e}")
            results.append(False)
    
    # Итоговый результат
    print("\n" + "="*50)
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*50)
    
    test_names = [
        "ConvBlock (основной)",
        "Отдельные компоненты", 
        "Поток градиентов",
        "Использование памяти"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        print(f"{i+1}. {name}: {status}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nОбщий результат: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! ConvBlock работает корректно!")
    else:
        print("⚠️  Некоторые тесты не пройдены. Проверьте код.")
    
    return passed == total


if __name__ == "__main__":
    main()