#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "NeuralNetworkComponent.generated.h"

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))

class DEEPLEARNING_PILOT_API UNeuralNetworkComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UNeuralNetworkComponent();

protected:
	virtual void BeginPlay() override;

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Neural Network")
	int32 InputSize = 17;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Neural Network")
	int32 Hidden1Size = 128;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Neural Network")
	int32 Hidden2Size = 128;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Neural Network")
	int32 OutputSize = 3;

	UFUNCTION(BlueprintCallable, Category = "Neural Network")
	void InitializeWeights();

	UFUNCTION(BlueprintCallable, Category = "Neural Network")
	TArray<float> FeedForward(const TArray<float>& Inputs);

	UFUNCTION(BlueprintCallable, Category = "Neural Network")
	void Mutate(float MutationRate = 0.05f, float MutationStrength = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "Neural Network")
	void CopyWeightsFrom(UNeuralNetworkComponent* SourceNetwork);

	UFUNCTION(BlueprintCallable, Category = "Neural Network")
	bool SaveWeightsToFile(const FString& FileName);

	UFUNCTION(BlueprintCallable, Category = "Neural Network")
	bool LoadWeightsFromFile(const FString& FileName);

private:
	TArray<float> WeightsIn_H1;
	TArray<float> Bias_H1;

	TArray<float> Weights_H1_H2;
	TArray<float> Bias_H2;

	TArray<float> Weights_H2_Out;
	TArray<float> Bias_Out;

	FORCEINLINE float ActivateTanh(float x) const
	{
		return FMath::Tanh(x);
	}

	FORCEINLINE float ActivateSigmoid(float x) const
	{
		return 1.0f / (1.0f + FMath::Exp(-FMath::Clamp(x, -50.0f, 50.0f)));
	}
};