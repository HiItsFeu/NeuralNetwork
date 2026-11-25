#include "NeuralNetworkComponent.h"

#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"
#include "Dom/JsonObject.h"

UNeuralNetworkComponent::UNeuralNetworkComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UNeuralNetworkComponent::BeginPlay()
{
	Super::BeginPlay();

	if (WeightsIn_H1.Num() == 0)
	{
		InitializeWeights();
	}
}

void UNeuralNetworkComponent::InitializeWeights()
{
	WeightsIn_H1.SetNum(InputSize * Hidden1Size);
	Bias_H1.SetNumZeroed(Hidden1Size);

	Weights_H1_H2.SetNum(Hidden1Size * Hidden2Size);
	Bias_H2.SetNumZeroed(Hidden2Size);

	Weights_H2_Out.SetNum(Hidden2Size * OutputSize);
	Bias_Out.SetNumZeroed(OutputSize);

	float Scale1 = FMath::Sqrt(1.0f / InputSize);
	for (float& W : WeightsIn_H1)
		W = FMath::FRandRange(-Scale1, Scale1);

	float Scale2 = FMath::Sqrt(1.0f / Hidden1Size);
	for (float& W : Weights_H1_H2)
		W = FMath::FRandRange(-Scale2, Scale2);

	float Scale3 = FMath::Sqrt(1.0f / Hidden2Size);
	for (float& W : Weights_H2_Out)
		W = FMath::FRandRange(-Scale3, Scale3);
}

TArray<float> UNeuralNetworkComponent::FeedForward(const TArray<float>& Inputs)
{
	TArray<float> Outputs;
	Outputs.SetNumZeroed(OutputSize);

	if (Inputs.Num() != InputSize)
	{
		UE_LOG(LogTemp, Error, TEXT("NeuralNetwork: Attendu %d inputs, reçu %d !"), InputSize, Inputs.Num());
		return Outputs;
	}

	TArray<float> Hidden1;
	Hidden1.SetNumUninitialized(Hidden1Size);
	for (int32 j = 0; j < Hidden1Size; ++j)
	{
		float Sum = Bias_H1[j];
		int32 RowOffset = j * InputSize;

		for (int32 i = 0; i < InputSize; ++i)
		{
			Sum += Inputs[i] * WeightsIn_H1[RowOffset + i];
		}
		Hidden1[j] = ActivateTanh(Sum);
	}

	TArray<float> Hidden2;
	Hidden2.SetNumUninitialized(Hidden2Size);
	for (int32 j = 0; j < Hidden2Size; ++j)
	{
		float Sum = Bias_H2[j];
		int32 RowOffset = j * Hidden1Size;

		for (int32 i = 0; i < Hidden1Size; ++i)
		{
			Sum += Hidden1[i] * Weights_H1_H2[RowOffset + i];
		}
		Hidden2[j] = ActivateTanh(Sum);
	}

	for (int32 j = 0; j < OutputSize; ++j)
	{
		float Sum = Bias_Out[j];
		int32 RowOffset = j * Hidden2Size;

		for (int32 i = 0; i < Hidden2Size; ++i)
		{
			Sum += Hidden2[i] * Weights_H2_Out[RowOffset + i];
		}

		if (j == 0)
		{
			Outputs[j] = ActivateTanh(Sum);
		}
		else
		{
			Outputs[j] = ActivateSigmoid(Sum);
		}
	}
	return Outputs;
}

void UNeuralNetworkComponent::Mutate(float MutationRate, float MutationStrength)
{
	auto MutateArray = [&](TArray<float>& Arr)
		{
			for (float& val : Arr)
			{
				if (FMath::FRand() < MutationRate)
				{
					val += FMath::FRandRange(-MutationStrength, MutationStrength);
				}
			}
		};

	MutateArray(WeightsIn_H1);
	MutateArray(Bias_H1);
	MutateArray(Weights_H1_H2);
	MutateArray(Bias_H2);
	MutateArray(Weights_H2_Out);
	MutateArray(Bias_Out);
}

void UNeuralNetworkComponent::CopyWeightsFrom(UNeuralNetworkComponent* SourceNetwork)
{
	if (!SourceNetwork) return;

	WeightsIn_H1 = SourceNetwork->WeightsIn_H1;
	Bias_H1 = SourceNetwork->Bias_H1;

	Weights_H1_H2 = SourceNetwork->Weights_H1_H2;
	Bias_H2 = SourceNetwork->Bias_H2;

	Weights_H2_Out = SourceNetwork->Weights_H2_Out;
	Bias_Out = SourceNetwork->Bias_Out;
}

bool UNeuralNetworkComponent::SaveWeightsToFile(const FString& FileName)
{
	TSharedPtr<FJsonObject> RootObj = MakeShareable(new FJsonObject());

	auto AddFloatArray = [&](const FString& Key, const TArray<float>& Arr)
		{
			TArray<TSharedPtr<FJsonValue>> JsonArr;
			for (float val : Arr)
			{
				JsonArr.Add(MakeShareable(new FJsonValueNumber(val)));
			}
			RootObj->SetArrayField(Key, JsonArr);
		};

	AddFloatArray(TEXT("Weights_In_H1"), WeightsIn_H1);
	AddFloatArray(TEXT("Bias_H1"), Bias_H1);
	AddFloatArray(TEXT("Weights_H1_H2"), Weights_H1_H2);
	AddFloatArray(TEXT("Bias_H2"), Bias_H2);
	AddFloatArray(TEXT("Weights_H2_Out"), Weights_H2_Out);
	AddFloatArray(TEXT("Bias_Out"), Bias_Out);

	FString OutputString;
	TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
	FJsonSerializer::Serialize(RootObj.ToSharedRef(), Writer);

	FString FullPath = FPaths::ProjectSavedDir() / TEXT("NeuralWeights") / (FileName + TEXT(".json"));
	return FFileHelper::SaveStringToFile(OutputString, *FullPath);
}

bool UNeuralNetworkComponent::LoadWeightsFromFile(const FString& FileName)
{
	FString FullPath = FPaths::ProjectSavedDir() / TEXT("NeuralWeights") / (FileName + TEXT(".json"));
	FString JsonString;

	if (!FFileHelper::LoadFileToString(JsonString, *FullPath)) return false;

	TSharedPtr<FJsonObject> RootObj;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);

	if (!FJsonSerializer::Deserialize(Reader, RootObj) || !RootObj.IsValid()) return false;

	auto ReadFloatArray = [&](const FString& Key, TArray<float>& OutArr)
		{
			const TArray<TSharedPtr<FJsonValue>>* JsonArr;
			if (RootObj->TryGetArrayField(Key, JsonArr))
			{
				OutArr.SetNum(JsonArr->Num());
				for (int32 i = 0; i < JsonArr->Num(); ++i)
					OutArr[i] = (*JsonArr)[i]->AsNumber();
			}
		};

	ReadFloatArray(TEXT("Weights_In_H1"), WeightsIn_H1);
	ReadFloatArray(TEXT("Bias_H1"), Bias_H1);
	ReadFloatArray(TEXT("Weights_H1_H2"), Weights_H1_H2);
	ReadFloatArray(TEXT("Bias_H2"), Bias_H2);
	ReadFloatArray(TEXT("Weights_H2_Out"), Weights_H2_Out);
	ReadFloatArray(TEXT("Bias_Out"), Bias_Out);

	return true;
}